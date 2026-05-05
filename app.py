import datetime
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

import numpy as np
import operator
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ── Configuración ──────────────────────────────────────────────────────────────

load_dotenv()
# Carga automática: GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
# Añadir más claves al .env / Streamlit secrets no requiere cambios en el código.
_base = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
GOOGLE_API_KEYS = [_base] if _base else []
_i = 2
while True:
    _k = st.secrets.get(f"GOOGLE_API_KEY_{_i}", os.getenv(f"GOOGLE_API_KEY_{_i}"))
    if not _k:
        break
    GOOGLE_API_KEYS.append(_k)
    _i += 1

CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "base_fiscal"
MAX_MESSAGES    = 10

BASE_DIR     = Path(__file__).parent
PRACTICOS_ES = BASE_DIR / "data/manuales/practicos/es"
WEB_ES       = BASE_DIR / "data/manuales/web/es"
LEYES_DIR    = BASE_DIR / "data/manuales/leyes"

MANUAL_METADATA = {
    (PRACTICOS_ES, "manual_iva_303_2025.pdf"):                  {"modelos": "303",     "perfil": "ambos",    "tipo": "manual_practico", "idioma": "es"},
    (PRACTICOS_ES, "manual_actividades_economicas_111_115.pdf"): {"modelos": "111,115", "perfil": "ambos",    "tipo": "manual_practico", "idioma": "es"},
    (PRACTICOS_ES, "manual_renta_100_130_2025_parte1.pdf"):      {"modelos": "100,130", "perfil": "autonomo", "tipo": "manual_practico", "idioma": "es"},
    (PRACTICOS_ES, "manual_renta_100_130_2025_parte2.pdf"):      {"modelos": "100,130", "perfil": "autonomo", "tipo": "manual_practico", "idioma": "es"},
    (PRACTICOS_ES, "manual_sociedades_200_202_2024.pdf"):        {"modelos": "200,202", "perfil": "sociedad", "tipo": "manual_practico", "idioma": "es"},
    (WEB_ES, "manual_rentaweb_100_2024.pdf"):                    {"modelos": "100",     "perfil": "autonomo", "tipo": "manual_web",      "idioma": "es"},
    (WEB_ES, "manual_sociedadesweb_200_2024.pdf"):               {"modelos": "200",     "perfil": "sociedad", "tipo": "manual_web",      "idioma": "es"},
}

SYSTEM_PROMPT = """Eres un asesor fiscal experto de una gestoría española llamada GestorIA.
Tu función es ayudar a gestores y clientes con las obligaciones fiscales de autónomos y sociedades en España.

## ROL Y LÍMITES

Eres un asistente especializado EXCLUSIVAMENTE en fiscalidad española. No respondas preguntas fuera de este ámbito.
Si te preguntan algo que no es fiscal (contabilidad general, derecho laboral, etc.), indica amablemente que está fuera de tu alcance.

## IDIOMA

Detecta el idioma en que escribe el usuario y responde siempre en ese mismo idioma.
El idioma de los documentos recuperados (contexto) NO influye en tu idioma de respuesta.

## FUENTES Y JERARQUÍA

Aplica siempre esta prioridad al responder:

1. **Contexto RAG** (documentos recuperados) — máxima autoridad para datos concretos: fechas, casillas, porcentajes, plazos, importes. Si el RAG y tu conocimiento general difieren, prevalece siempre el RAG.
2. **Conocimiento general como asesor fiscal** — solo para procedimientos estándar (acceso a sede AEAT, Cl@ve, certificado digital). Cita como "Procedimiento estándar AEAT".
3. **Ninguna fuente** — si no hay datos en ninguna de las dos fuentes anteriores, declara la ausencia explícitamente.

Si tienes información parcial, responde con lo que tengas y señala qué falta: "Sobre X dispongo de [dato], pero no tengo información sobre Y en mi base de conocimiento."
Solo cierra con "No dispongo de información suficiente..." cuando no tengas absolutamente ningún dato relevante.

## VIGENCIA DEL CONTEXTO

Si en los fragmentos RAG recuperados detectas referencias a ejercicios anteriores al trimestre actual (por ejemplo, menciones a "2023", "2024" o años anteriores en fechas de plazo o nombres de modelos), añade al final de tu respuesta: "⚠️ Parte del contexto recuperado puede corresponder a ejercicios anteriores. Verifica los datos en la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) antes de actuar."
No añadas este aviso si los fragmentos son coherentes con el ejercicio fiscal actual (2025–2026).

## IDENTIFICACIÓN DE PERFIL

- Identifica el perfil del cliente antes de responder: autónomo, sociedad, o ambos.
- Si el perfil aparece en la línea "Perfil del cliente" al inicio del mensaje, úsalo directamente sin volver a preguntar.
- Si el perfil NO está claro ni en esa línea ni en el historial, PREGUNTA antes de responder. No asumas.
- Una vez identificado, el perfil persiste durante toda la conversación. Inclúyelo en la primera respuesta y en aquellas donde aporte claridad (cambio de modelo, respuesta larga). En respuestas de seguimiento cortas puede omitirse si ya es evidente del contexto.
- En preguntas de seguimiento ("¿y el 130?", "¿cuánto tengo que pagar?"), usa el perfil y contexto del turno anterior sin solicitar aclaración si la pregunta es razonablemente interpretable.
- Si el modelo preguntado no aplica al perfil del cliente, indícalo antes de responder y redirige al modelo correcto cuando sea posible. Ejemplos: "El modelo 130 no aplica a sociedades — el equivalente es el modelo 202." / "El modelo 200 es exclusivo de sociedades — los autónomos liquidan el IRPF con el modelo 100."

## NIVEL TÉCNICO

Adapta el nivel de detalle según quién pregunta:
- **Gestor / asesor fiscal** — usa terminología técnica (casillas, regímenes, base imponible, devengo). Respuestas densas y precisas.
- **Cliente final** — lenguaje claro, sin jerga. Explica brevemente qué significa cada término técnico la primera vez que lo uses.

Si no está claro el tipo de interlocutor, usa un nivel intermedio: terminología técnica con una frase de contexto cuando sea necesario.

## ESTRUCTURA DE RESPUESTA

Adapta la estructura al tipo de pregunta. Incluye SOLO los apartados relevantes:

**Preguntas de plazo o calendario** → Perfil | Modelo + fecha límite + domiciliación + inicio preparación | Fuente
**Preguntas de cumplimentación o casillas** → Perfil | Nombre de la casilla + explicación técnica | Fuente
**Preguntas de procedimiento o pasos** → Perfil | Lista numerada de pasos | Fuente
**Preguntas de obligaciones generales** → Perfil | Lista de modelos aplicables con plazo e inicio preparación | Fuente
**Preguntas mixtas** → combina las secciones necesarias en orden lógico, sin duplicar información.

Nunca incluyas secciones vacías ni encabezados sin contenido. Si la pregunta solo pide un dato concreto (una fecha, una casilla), responde directamente.

## PLAZOS Y ANTELACIÓN

- La fecha de hoy y el trimestre activo aparecen en la línea "Fecha de hoy — Trimestre actual" del mensaje. Úsalos para resolver preguntas sin trimestre explícito ("¿qué tengo pendiente?", "¿el trimestre que viene?"). Si esa línea no está presente, deduce el trimestre a partir de tu conocimiento de la fecha actual: enero–marzo=1T, abril–junio=2T, julio–septiembre=3T, octubre–diciembre=4T.
- Días de preparación recomendados por tipo de obligación (úsalos si el contexto RAG no especifica otro valor):
  - Modelos trimestrales (303, 130, 111, 115, 202): 10 días antes del plazo
  - Modelos anuales simples (390, 347): 15 días antes del plazo
  - Modelos anuales complejos (100, 200): 30 días antes del plazo
- Cuando informes de un plazo, calcula y muestra siempre la fecha de inicio de preparación.
- Si el usuario pregunta "¿qué tengo pendiente?", lista TODAS las obligaciones del trimestre activo ordenadas por fecha límite.

## CORRECCIÓN DE ERRORES DEL USUARIO

Si detectas una incoherencia en la pregunta (trimestre incorrecto para ese modelo, fecha imposible, modelo que no aplica al perfil), corrígela de forma breve y directa antes de responder:
"El modelo 130 no tiene presentación en el cuarto trimestre — el último es en octubre (3T). Te respondo sobre el 3T:"

## FUENTES — CÓMO CITARLAS

Al citar la fuente usa el nombre descriptivo cuando mejore la credibilidad, no solo el nombre de fichero:
- `calendario_fiscal.csv` → "Calendario fiscal AEAT 2026"
- `obligaciones_perfil.csv` → "Mapa de obligaciones por perfil"
- `manual_iva_303_2025.pdf` → "Manual práctico IVA 303 (AEAT 2025)"
- `manual_renta_100_130_2025_parte1.pdf` / `parte2.pdf` → "Manual práctico Renta 100/130 (AEAT 2025)"
- `manual_sociedades_200_202_2024.pdf` → "Manual práctico Sociedades 200/202 (AEAT 2024)"
- `manual_actividades_economicas_111_115.pdf` → "Manual Actividades Económicas 111/115 (AEAT)"
- `manual_rentaweb_100_2024.pdf` → "Manual RentaWeb 100 (AEAT 2024)"
- `manual_sociedadesweb_200_2024.pdf` → "Manual SociedadesWeb 200 (AEAT 2024)"
- Conocimiento propio de procedimiento → "Procedimiento estándar AEAT"

## CONTEXTO ACUMULADO EN CONVERSACIÓN

Si el usuario hace varias preguntas encadenadas sobre el mismo modelo o tema, no repitas explicaciones ya dadas en el mismo hilo. Céntrate solo en la información nueva que aporta la pregunta actual. Si necesitas referirte a algo ya explicado, usa una referencia breve: "Como comenté antes, el plazo es el 20 de julio."

## TONO

Profesional y directo. Usa listas y negritas para facilitar la lectura.
Evita relleno vacío ("¡Claro!", "¡Por supuesto!") pero puedes usar una transición breve cuando el contexto lo pida ("En ese caso," "Para este perfil,").

---

## EJEMPLOS DE RESPUESTA CORRECTA

**Ejemplo 1 — Plazo de un modelo concreto:**
Usuario: "Soy autónomo, ¿cuándo presento el 303 del 2T?"

Respuesta:
**Perfil:** Autónomo.
**Modelo 303 — Autoliquidación IVA 2T 2026:**
- Fecha límite: 20 de julio de 2026
- Domiciliación hasta: 15 de julio de 2026
- Inicio de preparación recomendado: 10 de julio de 2026 (10 días antes)
*Fuente: Calendario fiscal AEAT 2026*

---

**Ejemplo 2 — Perfil no especificado:**
Usuario: "¿Cuándo tengo que presentar el modelo 303?"

Respuesta:
Para darte la información correcta, necesito saber tu perfil fiscal. ¿Eres autónomo o representas a una sociedad?

---

**Ejemplo 3 — Cómo rellenar una casilla:**
Usuario: "Soy autónomo. ¿Cómo relleno la casilla 01 del modelo 303?"

Respuesta:
**Perfil:** Autónomo.
**Casilla 01 — Base imponible al tipo general (21%):**
Incluye el importe total de las entregas de bienes y prestaciones de servicios sujetas y no exentas de IVA gravadas al 21%, sin incluir el propio impuesto.
*Fuente: Manual práctico IVA 303 (AEAT 2025)*

---

**Ejemplo 4 — Obligaciones de un trimestre:**
Usuario: "Somos una S.L. ¿Qué tenemos que presentar en el segundo trimestre?"

Respuesta:
**Perfil:** Sociedad.
**Obligaciones 2T 2026:**
- Modelo 303 — IVA 2T | Plazo: 20 julio | Inicio preparación: 10 julio
- Modelo 111 — Retenciones IRPF 2T | Plazo: 20 julio | Inicio preparación: 10 julio
- Modelo 115 — Retenciones alquileres 2T | Plazo: 20 julio | Inicio preparación: 10 julio
- Modelo 202 — Pago fraccionado IS | Plazo: 20 julio | Inicio preparación: 10 julio
*Fuente: Calendario fiscal AEAT 2026, Mapa de obligaciones por perfil*

---

**Ejemplo 5 — Información no disponible:**
Usuario: "¿Cuál es el tipo de recargo de equivalencia para el aceite de oliva?"

Respuesta:
No dispongo de información suficiente sobre este punto en mi base de conocimiento. Te recomiendo consultar la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) o al gestor responsable.

---

**Ejemplo 6 — Pasos para presentar un modelo:**
Usuario: "Represento una sociedad. ¿Qué pasos debo seguir para presentar el IVA?"

Respuesta:
**Perfil:** Sociedad.
**Pasos para presentar el Modelo 303 — Autoliquidación IVA:**

1. **Recopilar la documentación** — Facturas emitidas (IVA repercutido) y facturas recibidas (IVA soportado) del trimestre.
2. **Calcular el resultado** — IVA repercutido − IVA soportado deducible. Si es positivo, hay que ingresar; si es negativo, se puede compensar o solicitar devolución.
3. **Acceder a la Sede Electrónica de la AEAT** — Con certificado digital, DNIe o Cl@ve PIN en sede.agenciatributaria.gob.es.
4. **Cumplimentar el formulario** — Ir a "Trámites destacados > Modelo 303". Rellenar casillas 01-09 (IVA devengado) y 28-44 (IVA deducible).
5. **Presentar y pagar si procede** — Se puede domiciliar hasta 5 días antes del plazo límite.
6. **Guardar el justificante** — La AEAT emite un CSV de confirmación como acuse de recibo oficial.

*Fuente: Manual práctico IVA 303 (AEAT 2025), procedimiento estándar AEAT*

---

**Ejemplo 7 — Pregunta de seguimiento (perfil ya conocido):**
[Turno anterior: el usuario dijo ser autónomo y preguntó por el modelo 303]
Usuario: "¿Y el 130?"

Respuesta:
**Modelo 130 — Pago fraccionado IRPF 2T 2026** (Autónomo, estimación directa):
- Fecha límite: 20 de julio de 2026
- Inicio de preparación recomendado: 10 de julio de 2026
*Fuente: Calendario fiscal AEAT 2026*

---

**Ejemplo 8 — Información parcialmente disponible:**
Usuario: "Soy autónomo. ¿Cuánto tengo que ingresar en el modelo 130 y cuál es la penalización por presentarlo tarde?"

Respuesta:
**Perfil:** Autónomo.
**Modelo 130 — Cálculo del importe:** El importe es el 20% del rendimiento neto de actividades económicas del trimestre, menos retenciones soportadas y pagos fraccionados anteriores del ejercicio.
*Fuente: Manual práctico Renta 100/130 (AEAT 2025)*

Sobre las penalizaciones por presentación fuera de plazo no dispongo de información en mi base de conocimiento. Te recomiendo consultar la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) para conocer el régimen de recargos aplicable.

---

**Ejemplo 9 — Corrección de error del usuario:**
Usuario: "Soy autónomo, ¿cuándo presento el modelo 130 del cuarto trimestre?"

Respuesta:
El modelo 130 no tiene presentación en el cuarto trimestre — el último pago fraccionado del año es el 3T (octubre). En enero se presenta el modelo 100 con la liquidación anual del IRPF.

**Modelo 130 — Pago fraccionado IRPF 3T 2026** (Autónomo):
- Fecha límite: 20 de octubre de 2026
- Inicio de preparación recomendado: 10 de octubre de 2026
*Fuente: Calendario fiscal AEAT 2026*

---

**Ejemplo 10 — Pregunta mixta (plazo + procedimiento):**
Usuario: "Soy autónomo. ¿Cuándo tengo que presentar el 303 del 1T y cómo lo hago?"

Respuesta:
**Perfil:** Autónomo.

**Plazo — Modelo 303 IVA 1T 2026:**
- Fecha límite: 20 de abril de 2026
- Domiciliación hasta: 15 de abril de 2026
- Inicio de preparación recomendado: 10 de abril de 2026
*Fuente: Calendario fiscal AEAT 2026*

**Cómo presentarlo:**
1. Accede a sede.agenciatributaria.gob.es con certificado digital o Cl@ve PIN.
2. Ve a "Trámites destacados > Modelo 303" y cumplimenta las casillas con los datos del trimestre.
3. Presenta y, si sale a ingresar, puedes domiciliar el pago hasta el 15 de abril.
4. Guarda el justificante con el CSV de confirmación.
*Fuente: procedimiento estándar AEAT*

---

**Ejemplo 11 — Modelo no aplicable al perfil:**
Usuario: "Soy autónomo, ¿cuándo presento el modelo 202?"

Respuesta:
El modelo 202 (Pagos fraccionados del Impuesto sobre Sociedades) no aplica a autónomos — es exclusivo de sociedades. Como autónomo, el pago fraccionado del IRPF se realiza con el **modelo 130** (estimación directa) o el **modelo 131** (estimación objetiva/módulos).

**Modelo 130 — Pago fraccionado IRPF 2T 2026** (Autónomo):
- Fecha límite: 20 de julio de 2026
- Inicio de preparación recomendado: 10 de julio de 2026
*Fuente: Calendario fiscal AEAT 2026, Mapa de obligaciones por perfil*
"""

# ── Moderación en cascada ──────────────────────────────────────────────────────
# Capas: Reglas (gratis) → ML (rápido) → LLM (solo ambiguos)

UMBRAL_CONFIANZA_ML = 0.85

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

@dataclass
class ResultadoModeracion:
    decision:  str    # "fiscal" | "offtopic"
    capa:      str    # "reglas" | "ml" | "llm"
    confianza: float

@st.cache_resource(show_spinner=False)
def _construir_clasificador_ml():
    ejemplos = [
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
    X, y = zip(*ejemplos)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(list(X), list(y))
    return clf

def moderar_pregunta(texto: str, llm_lite) -> ResultadoModeracion:
    # Capa 1: reglas
    if _KEYWORDS_FISCAL.search(texto):
        return ResultadoModeracion("fiscal",   "reglas", 1.0)
    if _KEYWORDS_OFFTOPIC.search(texto):
        return ResultadoModeracion("offtopic", "reglas", 1.0)
    # Capa 2: ML
    clf       = _construir_clasificador_ml()
    proba     = clf.predict_proba([texto])[0]
    idx_max   = int(np.argmax(proba))
    confianza = float(proba[idx_max])
    if confianza >= UMBRAL_CONFIANZA_ML:
        return ResultadoModeracion(clf.classes_[idx_max], "ml", confianza)
    # Capa 3: modelo lite para casos ambiguos
    raw = _extraer_texto(_invoke_con_retry(llm_lite, [HumanMessage(
        content="Clasifica esta pregunta como 'fiscal' o 'offtopic'. Responde SOLO con una palabra.\n\nPregunta: " + texto
    )], tipo="lite")).strip().lower()
    return ResultadoModeracion("fiscal" if "fiscal" in raw else "offtopic", "llm", 0.6)


# ── LLM-as-Judge ──────────────────────────────────────────────────────────────

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

def evaluar_respuesta(pregunta: str, respuesta: str, llm_lite) -> Optional[dict]:
    criterios = "precisión de fechas y plazos, claridad de la explicación, completitud según el perfil del usuario"
    try:
        prompt_messages = _PROMPT_JUEZ.format_messages(
            pregunta=pregunta, respuesta=respuesta, criterios=criterios
        )
        raw = _extraer_texto(_invoke_con_retry(llm_lite, prompt_messages, tipo="lite"))
        return json.loads(raw.strip().replace("```json", "").replace("```", "").strip())
    except Exception:
        return None


# ── Indexación de documentos ───────────────────────────────────────────────────

def _indexar_documentos(embeddings):
    import pdfplumber
    from langchain_core.documents import Document
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    semantic_splitter  = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    def cargar_pdfs(metadata_map):
        docs = []
        for (directorio, filename), meta in metadata_map.items():
            path = directorio / filename
            if not path.exists():
                continue
            textos = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        textos.append(t)
            doc_base = Document(page_content="\n\n".join(textos), metadata={**meta, "fuente": filename})
            try:
                chunks = semantic_splitter.split_documents([doc_base])
                if not chunks:
                    raise ValueError("0 chunks")
            except Exception:
                chunks = fallback_splitter.split_documents([doc_base])
            docs.extend(chunks)
        return docs

    def cargar_csv(path, tipo, sep=","):
        df = pd.read_csv(path, sep=sep)
        docs = []
        for _, row in df.iterrows():
            contenido = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            meta = {"fuente": path.name, "tipo": tipo}
            if "modelo"    in row: meta["modelos"]   = str(row["modelo"])
            if "perfil"    in row: meta["perfil"]    = str(row["perfil"])
            if "trimestre" in row: meta["trimestre"] = str(row["trimestre"])
            docs.append(Document(page_content=contenido, metadata=meta))
        return docs

    # En Streamlit Cloud (1 GB RAM) excluimos las leyes completas para no agotar memoria.
    all_docs = (
        cargar_pdfs(MANUAL_METADATA)
        + cargar_csv(BASE_DIR / "data/calendario_fiscal.csv",   "calendario",          sep=",")
        + cargar_csv(BASE_DIR / "data/obligaciones_perfil.csv", "obligaciones_perfil", sep=";")
    )
    return Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


# ── Agente LangGraph (cacheado) ────────────────────────────────────────────────

_KEYWORDS_PLAZOS = re.compile(
    r"\b(plazo|fecha|cuando|cuándo|vencimiento|trimestre|domicili|"
    r"antelacion|antelación|pendiente)\b",
    re.IGNORECASE,
)
_KEYWORDS_DOCS = re.compile(
    r"\b(casilla|rellenar|cumplimentar|calcul|base imponible|deduccion|deducción|"
    r"como se|cómo se|instruccion|instrucción|apartado|anexo|"
    r"paso|pasos|proceso|procedimiento|"
    r"c[oó]mo presento|c[oó]mo se presenta|c[oó]mo funciona|c[oó]mo hago|"
    r"c[oó]mo debo|qué pasos|qu[eé] debo hacer)\b",
    re.IGNORECASE,
)

MAX_RETRIES_RPM = 5

# Matriz de fallback: el agente usa modelos potentes primero; lite/judge usa modelos ligeros primero.
MODELOS_AGENTE = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite-preview",
]
MODELOS_LITE = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]

def _extraer_texto(msg) -> str:
    """Extrae texto de un mensaje cuyo .content puede ser str o list (Gemini 3+)."""
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        partes = [p.get("text", "") if isinstance(p, dict) else str(p) for p in c]
        return " ".join(partes)
    return str(c)

def _extraer_retry_delay(error_str: str, default: float = 15.0) -> float:
    match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)\s*s", error_str)
    return float(match.group(1)) + 1 if match else default

def _es_limite_diario(err: str) -> bool:
    return "GenerateRequestsPerDayPerProjectPerModel" in err

def _es_limite_rpm(err: str) -> bool:
    return "GenerateRequestsPerMinutePerProjectPerModel" in err

def _crear_llm(modelo: str, clave_idx: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=GOOGLE_API_KEYS[clave_idx],
        temperature=0,
    )

# Estado global: índice de modelo y clave activos para agente y lite
_estado = {
    "agente": {"modelo_idx": 0, "clave_idx": 0},
    "lite":   {"modelo_idx": 0, "clave_idx": 0},
}

def _siguiente_combinacion(tipo: str) -> bool:
    """Avanza a la siguiente combinación modelo+clave. Devuelve False si se agotaron todas."""
    modelos = MODELOS_AGENTE if tipo == "agente" else MODELOS_LITE
    est = _estado[tipo]
    # Primero rotar clave dentro del mismo modelo
    if est["clave_idx"] + 1 < len(GOOGLE_API_KEYS):
        est["clave_idx"] += 1
        return True
    # Si se agotaron las claves, bajar al siguiente modelo y resetear claves
    if est["modelo_idx"] + 1 < len(modelos):
        est["modelo_idx"] += 1
        est["clave_idx"] = 0
        return True
    return False

def _llm_actual(tipo: str) -> ChatGoogleGenerativeAI:
    modelos = MODELOS_AGENTE if tipo == "agente" else MODELOS_LITE
    est = _estado[tipo]
    return _crear_llm(modelos[est["modelo_idx"]], est["clave_idx"])

def _invoke_con_retry(llm_obj, messages: list, tipo: str = "agente"):
    """Invoca el LLM con retry RPM y fallback modelo+clave ante límite diario."""
    modelos = MODELOS_AGENTE if tipo == "agente" else MODELOS_LITE
    for _ in range(MAX_RETRIES_RPM + len(GOOGLE_API_KEYS) * len(modelos)):
        try:
            return llm_obj.invoke(messages)
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" not in err and "429" not in err:
                raise
            if _es_limite_rpm(err):
                delay = _extraer_retry_delay(err)
                time.sleep(delay)
                continue
            if _es_limite_diario(err):
                est = _estado[tipo]
                modelo_actual = modelos[est["modelo_idx"]]
                if not _siguiente_combinacion(tipo):
                    raise RuntimeError("Todas las combinaciones modelo+clave están agotadas.") from e
                nuevo_modelo = modelos[_estado[tipo]["modelo_idx"]]
                nueva_clave  = _estado[tipo]["clave_idx"] + 1
                st.warning(f"[{modelo_actual}] clave {est['clave_idx']} agotada → {nuevo_modelo} clave {nueva_clave}")
                llm_obj = _llm_actual(tipo)
                continue
            raise
    raise RuntimeError("Se agotaron los reintentos de la API.")

def _inicializar_llm(tipo: str = "agente") -> ChatGoogleGenerativeAI:
    """Inicializa el LLM con la primera combinación modelo+clave sin llamada de prueba.
    El fallback actúa en el primer uso real si la combinación está agotada."""
    if not GOOGLE_API_KEYS:
        raise RuntimeError("No hay ninguna GOOGLE_API_KEY configurada.")
    modelos = MODELOS_AGENTE if tipo == "agente" else MODELOS_LITE
    _estado[tipo]["modelo_idx"] = 0
    _estado[tipo]["clave_idx"]  = 0
    return _crear_llm(modelos[0], 0)


@st.cache_resource(show_spinner="Cargando base de conocimiento fiscal...")
def cargar_recursos():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    if Path(CHROMA_DIR).exists():
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    else:
        with st.spinner("Primera ejecución: indexando documentos (puede tardar unos minutos)..."):
            vectorstore = _indexar_documentos(embeddings)

    llm      = _inicializar_llm("agente")
    llm_lite = _inicializar_llm("lite")

    class AgentState(TypedDict):
        messages:      Annotated[list, operator.add]
        perfil:        str
        contexto_rag:  str
        tipo_consulta: str

    # ── Nodo: poda de historial ──
    def podar_historial(state: AgentState) -> AgentState:
        mensajes = state["messages"]
        if len(mensajes) <= MAX_MESSAGES:
            return {}
        n = len(mensajes) - MAX_MESSAGES
        if n % 2 != 0:
            n += 1
        return {"messages": [RemoveMessage(id=m.id) for m in mensajes[:n]]}

    # ── Nodo: clasificación de consulta ──
    def clasificar_consulta(state: AgentState) -> AgentState:
        ultima = state["messages"][-1].content
        if _KEYWORDS_DOCS.search(ultima):
            tipo = "documentos"
        elif _KEYWORDS_PLAZOS.search(ultima):
            tipo = "plazos"
        else:
            tipo = "general"
        return {"tipo_consulta": tipo}

    def router(state: AgentState) -> Literal["recuperar_plazos", "recuperar_documentos", "recuperar_general"]:
        return {
            "plazos":     "recuperar_plazos",
            "documentos": "recuperar_documentos",
            "general":    "recuperar_general",
        }[state["tipo_consulta"]]

    # ── Utilidad: combinar docs sin duplicados ──
    def _combinar(docs_a, docs_b):
        vistos, resultado = set(), []
        for doc in docs_a + docs_b:
            clave = doc.page_content[:100]
            if clave not in vistos:
                vistos.add(clave)
                resultado.append(doc)
        return "\n\n".join(
            f"[{d.metadata.get('fuente','?')}]\n{d.page_content}" for d in resultado
        )

    # ── Nodos de recuperación RAG especializados ──
    def recuperar_plazos(state: AgentState) -> AgentState:
        ultima = state["messages"][-1].content
        perfil = state.get("perfil", "")
        docs_csv = vectorstore.as_retriever(search_kwargs={
            "k": 10, "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
        }).invoke(ultima)
        kw = {"k": 3}
        if perfil in ("autonomo", "sociedad"):
            kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
        docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)
        return {"contexto_rag": _combinar(docs_csv, docs_m)}

    def recuperar_documentos(state: AgentState) -> AgentState:
        ultima = state["messages"][-1].content
        perfil = state.get("perfil", "")
        kw = {"k": 12}
        if perfil in ("autonomo", "sociedad"):
            kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
        docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)
        docs_csv = vectorstore.as_retriever(search_kwargs={
            "k": 3, "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
        }).invoke(ultima)
        return {"contexto_rag": _combinar(docs_m, docs_csv)}

    def recuperar_general(state: AgentState) -> AgentState:
        ultima = state["messages"][-1].content
        perfil = state.get("perfil", "")
        kw = {"k": 5}
        if perfil in ("autonomo", "sociedad"):
            kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
        docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)
        docs_csv = vectorstore.as_retriever(search_kwargs={
            "k": 6, "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
        }).invoke(ultima)
        return {"contexto_rag": _combinar(docs_m, docs_csv)}

    # ── Nodo: generación de respuesta ──
    def generar_respuesta(state: AgentState) -> AgentState:
        contexto  = state.get("contexto_rag", "")
        historial = state["messages"]

        hoy = datetime.date.today()
        trimestre = (hoy.month - 1) // 3 + 1
        contexto_temporal = f"Fecha de hoy: {hoy.strftime('%d/%m/%Y')} — Trimestre actual: {trimestre}T 2026\n"

        perfil_actual = state.get("perfil", "")
        perfil_linea  = ""
        if perfil_actual:
            label = {"autonomo": "Autónomo", "sociedad": "Sociedad"}.get(perfil_actual, "")
            perfil_linea = f"Perfil del cliente: {label}\n"

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages += [m for m in historial[:-1] if not isinstance(m, RemoveMessage)]
        messages.append(HumanMessage(content=(
            f"{contexto_temporal}{perfil_linea}"
            f"Contexto recuperado de la base de conocimiento:\n---\n{contexto}\n---\n\n"
            f"Pregunta del usuario: {historial[-1].content}"
        )))
        respuesta = _invoke_con_retry(llm, messages, tipo="agente")

        perfil_actual = state.get("perfil", "")
        if not perfil_actual:
            # Buscar perfil en todo el historial, del más reciente al más antiguo
            _KW_AUTO = re.compile(r"\baut[oó]nomo\b", re.IGNORECASE)
            _KW_SOC  = re.compile(r"\b(sociedad|empresa|s\.l|s\.a)\b", re.IGNORECASE)
            for msg in reversed(historial):
                texto = msg.content.lower()
                if _KW_AUTO.search(texto):
                    perfil_actual = "autonomo"
                    break
                if _KW_SOC.search(texto):
                    perfil_actual = "sociedad"
                    break

        return {"messages": [AIMessage(content=respuesta.content)], "perfil": perfil_actual}

    # ── Construcción del grafo ──
    workflow = StateGraph(AgentState)
    workflow.add_node("podar_historial",      podar_historial)
    workflow.add_node("clasificar_consulta",  clasificar_consulta)
    workflow.add_node("recuperar_plazos",     recuperar_plazos)
    workflow.add_node("recuperar_documentos", recuperar_documentos)
    workflow.add_node("recuperar_general",    recuperar_general)
    workflow.add_node("generar_respuesta",    generar_respuesta)

    workflow.add_edge(START, "podar_historial")
    workflow.add_edge("podar_historial", "clasificar_consulta")
    workflow.add_conditional_edges(
        "clasificar_consulta", router,
        {
            "recuperar_plazos":     "recuperar_plazos",
            "recuperar_documentos": "recuperar_documentos",
            "recuperar_general":    "recuperar_general",
        },
    )
    workflow.add_edge("recuperar_plazos",     "generar_respuesta")
    workflow.add_edge("recuperar_documentos", "generar_respuesta")
    workflow.add_edge("recuperar_general",    "generar_respuesta")
    workflow.add_edge("generar_respuesta",    END)

    agente = workflow.compile(checkpointer=MemorySaver())
    return agente, llm, llm_lite


# ── Interfaz Streamlit ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asistente Fiscal — GestorIA",
    page_icon="📋",
    layout="centered",
)

st.title("📋 Asistente Fiscal GestorIA")
st.caption("Asesor fiscal para autónomos y sociedades en España · Powered by Gemini + RAG")

tab_chat, tab_calendario = st.tabs(["💬 Asistente", "📅 Calendario Fiscal"])

with st.sidebar:
    st.header("Configuración")
    perfil_opcion = st.selectbox(
        "Perfil del cliente",
        options=["Sin especificar", "Autónomo", "Sociedad"],
        index=0,
    )
    perfil_map          = {"Sin especificar": "", "Autónomo": "autonomo", "Sociedad": "sociedad"}
    perfil_seleccionado = perfil_map[perfil_opcion]

    mostrar_evaluacion = st.toggle("Mostrar evaluación de respuestas", value=False)

    if st.button("🔄 Nueva conversación"):
        st.session_state.messages    = []
        st.session_state.agent_state = {
            "messages": [], "perfil": perfil_seleccionado, "contexto_rag": "", "tipo_consulta": "",
        }
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("**Modelos cubiertos:**")
    st.markdown("303 · 130 · 111 · 115 · 100 · 200 · 202 · 390 · 347")

# Inicializar estado de sesión
if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "thread_id"   not in st.session_state:
    st.session_state.thread_id   = str(uuid.uuid4())
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [], "perfil": perfil_seleccionado, "contexto_rag": "", "tipo_consulta": "",
    }

if st.session_state.agent_state.get("perfil") != perfil_seleccionado and not st.session_state.messages:
    st.session_state.agent_state["perfil"] = perfil_seleccionado

agente, llm, llm_lite = cargar_recursos()

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evaluacion") and mostrar_evaluacion:
                ev = msg["evaluacion"]
                with st.expander(f"Evaluación — {ev.get('puntuacion_global', '-')}/10"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Precisión",   f"{ev.get('precision_tecnica', '-')}/10")
                    c2.metric("Claridad",    f"{ev.get('claridad', '-')}/10")
                    c3.metric("Completitud", f"{ev.get('completitud', '-')}/10")
                    st.caption(ev.get("justificacion", ""))

    pregunta = st.chat_input("Escribe tu pregunta fiscal...")

    if pregunta and pregunta.strip():
        moderacion = moderar_pregunta(pregunta, llm_lite)

        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        if moderacion.decision == "offtopic":
            respuesta = (
                "Esta pregunta está fuera de mi ámbito de especialización. "
                "Soy un asistente fiscal especializado en obligaciones tributarias españolas. "
                "¿Puedo ayudarte con alguna consulta fiscal?"
            )
            with st.chat_message("assistant"):
                st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Consultando la base de conocimiento..."):
                    state  = st.session_state.agent_state
                    state["messages"] = state.get("messages", []) + [HumanMessage(content=pregunta)]
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = agente.invoke(state, config=config)
                    st.session_state.agent_state = result
                    respuesta = _extraer_texto(result["messages"][-1])
                st.markdown(respuesta)

                evaluacion = None
                if mostrar_evaluacion:
                    with st.spinner("Evaluando calidad..."):
                        evaluacion = evaluar_respuesta(pregunta, respuesta, llm_lite)
                    if evaluacion:
                        with st.expander(f"Evaluación — {evaluacion.get('puntuacion_global', '-')}/10"):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Precisión",   f"{evaluacion.get('precision_tecnica', '-')}/10")
                            c2.metric("Claridad",    f"{evaluacion.get('claridad', '-')}/10")
                            c3.metric("Completitud", f"{evaluacion.get('completitud', '-')}/10")
                            st.caption(evaluacion.get("justificacion", ""))

            st.session_state.messages.append({
                "role": "assistant", "content": respuesta, "evaluacion": evaluacion,
            })

        st.rerun()

with tab_calendario:
    st.subheader("Calendario de obligaciones fiscales 2026")

    @st.cache_data
    def cargar_calendario():
        df = pd.read_csv(BASE_DIR / "data/calendario_fiscal.csv")
        df["fecha_limite_2026"]   = pd.to_datetime(df["fecha_limite_2026"])
        df["domiciliacion_hasta"] = pd.to_datetime(df["domiciliacion_hasta"], errors="coerce")
        return df.sort_values("fecha_limite_2026")

    df_cal = cargar_calendario()

    opciones_perfil = {"Todos": None, "Autónomo": "autonomo", "Sociedad": "sociedad"}
    perfil_filtro   = st.radio(
        "Mostrar obligaciones de:",
        options=list(opciones_perfil.keys()),
        index={"": 0, "autonomo": 1, "sociedad": 2}.get(perfil_seleccionado, 0),
        horizontal=True,
    )
    valor_filtro = opciones_perfil[perfil_filtro]
    df_vis = df_cal[df_cal["perfil"].isin([valor_filtro, "ambos"])].copy() if valor_filtro else df_cal.copy()

    hoy = datetime.date.today()
    df_vis["_dias"] = (df_vis["fecha_limite_2026"].dt.date - hoy).apply(lambda d: d.days)
    df_vis = df_vis.reset_index(drop=True)

    def estado(d):
        if d < 0:   return "✅ Vencido"
        if d <= 7:  return "🔴 ≤ 7 días"
        if d <= 30: return "🟡 ≤ 30 días"
        return "🟢 Pendiente"

    st.dataframe(pd.DataFrame({
        "Estado":              df_vis["_dias"].apply(estado),
        "Modelo":              df_vis["modelo"],
        "Obligación":          df_vis["nombre"],
        "Perfil":              df_vis["perfil"],
        "Fecha límite":        df_vis["fecha_limite_2026"].dt.strftime("%d/%m/%Y"),
        "Domiciliación hasta": df_vis["domiciliacion_hasta"].apply(
            lambda x: x.strftime("%d/%m/%Y") if pd.notna(x) else "—"
        ),
        "Días preparación":    df_vis["dias_preparacion_recomendados"],
        "Periodicidad":        df_vis["periodicidad"],
    }), use_container_width=True, hide_index=True)
