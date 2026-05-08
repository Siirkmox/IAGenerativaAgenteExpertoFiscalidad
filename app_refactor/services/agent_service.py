import json
import re
from typing import Optional

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agents.graph import construir_grafo
from core.config import UMBRAL_CONFIANZA_ML
from models.schemas import ResultadoModeracion
from services.llm_service import (
    _extraer_texto,
    inicializar_llm,
    invoke_con_retry,
)
from utils.logger import logger


# ── Inicialización de LLMs ─────────────────────────────────────────────────────
llm      = inicializar_llm("agente")
llm_lite = inicializar_llm("lite")

# ── Grafo del agente ───────────────────────────────────────────────────────────
agente = construir_grafo(llm)


# ── Moderación en cascada ──────────────────────────────────────────────────────
# Capas: Reglas (gratis) → ML (rápido) → LLM (solo para casos ambiguos)
# El objetivo es no gastar cuota de API en preguntas claramente fuera de ámbito.

_KEYWORDS_FISCAL = re.compile(
    r"\b(modelo|irpf|iva|impuesto|declaraci[oó]n|renta|hacienda|aeat|tribut|fiscal|"
    r"autonomo|aut[oó]nomo|sociedad|empresa|s\.l|factura|casilla|plazo|trimestre|"
    r"303|130|111|115|100|200|202|347|390|retenci[oó]n|deducci[oó]n)\b",
    re.IGNORECASE,
)
_KEYWORDS_OFFTOPIC = re.compile(
    r"\b(receta|cocina|deporte|f[uú]tbol|pel[ií]cula|m[uú]sica|viaje|hotel|"
    r"tiempo|clima|meteorolog[ií]a|amor|relaci[oó]n|juego|videojuego)\b",
    re.IGNORECASE,
)

# Clasificador ML entrenado con ejemplos mínimos: TF-IDF + Regresión Logística.
# Es ligero, sin coste de API, y resuelve la mayoría de casos ambiguos.
_ejemplos_ml = [
    ("¿Cuándo presento el modelo 303?",                    "fiscal"),
    ("¿Qué obligaciones tengo como autónomo?",             "fiscal"),
    ("¿Cómo se rellena la casilla 01 del IVA?",           "fiscal"),
    ("Plazo para presentar el IRPF 2025",                  "fiscal"),
    ("¿Qué es la domiciliación en el modelo 130?",         "fiscal"),
    ("Retenciones en el modelo 111 segundo trimestre",     "fiscal"),
    ("¿Cuánto tiempo tengo para presentar el IS?",         "fiscal"),
    ("Deducciones en el modelo 303",                       "fiscal"),
    ("¿Cómo me doy de alta como autónomo en hacienda?",    "fiscal"),
    ("¿Qué es el pago fraccionado del IRPF?",              "fiscal"),
    ("¿Cuál es la mejor receta de paella?",                "offtopic"),
    ("¿Quién ganó el partido de ayer?",                    "offtopic"),
    ("Recomiéndame una película de terror",                "offtopic"),
    ("¿Qué tiempo hace en Madrid?",                        "offtopic"),
    ("¿Cómo se llama el presidente de Francia?",           "offtopic"),
    ("Cuéntame un chiste",                                 "offtopic"),
    ("¿Cuál es la capital de Australia?",                  "offtopic"),
    ("Dame una ruta de senderismo",                        "offtopic"),
]
_X, _y = zip(*_ejemplos_ml)
_clasificador_ml = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
])
_clasificador_ml.fit(list(_X), list(_y))


def moderar_pregunta(texto: str) -> ResultadoModeracion:
    # Capa 1: reglas basadas en keywords (coste cero)
    if _KEYWORDS_FISCAL.search(texto):
        return ResultadoModeracion("fiscal",   "reglas", 1.0)
    if _KEYWORDS_OFFTOPIC.search(texto):
        return ResultadoModeracion("offtopic", "reglas", 1.0)

    # Capa 2: clasificador ML (coste cero, <1ms)
    proba     = _clasificador_ml.predict_proba([texto])[0]
    idx_max   = int(np.argmax(proba))
    confianza = float(proba[idx_max])
    if confianza >= UMBRAL_CONFIANZA_ML:
        return ResultadoModeracion(_clasificador_ml.classes_[idx_max], "ml", confianza)

    # Capa 3: LLM ligero solo para casos ambiguos que no han pasado por los filtros anteriores
    raw = _extraer_texto(invoke_con_retry(llm_lite, [HumanMessage(
        content="Clasifica esta pregunta como 'fiscal' o 'offtopic'. Responde SOLO con una palabra.\n\nPregunta: " + texto
    )], tipo="lite")).strip().lower()
    return ResultadoModeracion("fiscal" if "fiscal" in raw else "offtopic", "llm", 0.6)


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────
# Evalúa la calidad de cada respuesta en tres dimensiones usando el modelo lite.
_PROMPT_JUEZ = ChatPromptTemplate.from_template("""
Eres un evaluador experto en asesoría fiscal española.
Evalúa la calidad de esta respuesta según los criterios dados.

**Pregunta:** {pregunta}
**Respuesta a evaluar:** {respuesta}
**Criterios:** {criterios}

Responde ÚNICAMENTE con este JSON (sin markdown):
{{
  "puntuacion_global": número entre 1 y 10,
  "precision_tecnica": número entre 1 y 10,
  "claridad": número entre 1 y 10,
  "completitud": número entre 1 y 10,
  "justificacion": "máximo 2 oraciones"
}}
""")

def evaluar_respuesta(pregunta: str, respuesta: str) -> Optional[dict]:
    criterios = "precisión de fechas y plazos, claridad de la explicación, completitud según el perfil del usuario"
    try:
        prompt_messages = _PROMPT_JUEZ.format_messages(
            pregunta=pregunta, respuesta=respuesta, criterios=criterios
        )
        raw = _extraer_texto(invoke_con_retry(llm_lite, prompt_messages, tipo="lite"))
        return json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        logger.warning(f"Error en evaluación LLM-as-Judge: {e}")
        return None


# ── Función principal de ejecución ─────────────────────────────────────────────
def run_agent(state: dict, thread_id: str) -> dict:
    """Invoca el grafo con el estado actual y devuelve el estado resultante."""
    config = {"configurable": {"thread_id": thread_id}}
    return agente.invoke(state, config=config)
