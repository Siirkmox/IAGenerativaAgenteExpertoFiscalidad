import datetime
import json
import os
import re
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
GOOGLE_API_KEY  = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
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

## REGLA DE ORO: NO ALUCINACIONES

- Para datos concretos y verificables (fechas, casillas, porcentajes, plazos exactos): usa ÚNICAMENTE lo que aparece en el contexto proporcionado. Si no está, responde: "No dispongo de información suficiente sobre este punto en mi base de conocimiento. Te recomiendo consultar la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) o al gestor responsable."
- Para procedimientos y pasos generales de presentación telemática (acceso a Cl@ve, certificado digital, sede AEAT): puedes usar tu conocimiento como asesor fiscal, pero cita siempre la fuente ("Procedimiento estándar AEAT") y señala si el contexto RAG aporta información adicional.

## IDENTIFICACIÓN DE PERFIL

- SIEMPRE identifica el perfil del cliente antes de responder: autónomo, sociedad, o ambos.
- Si el perfil NO está claro en la pregunta ni en el historial, PREGUNTA antes de dar cualquier información fiscal. No asumas.
- Una vez identificado el perfil, recuérdalo durante toda la conversación. No vuelvas a preguntarlo.

## ESTRUCTURA DE RESPUESTA

Organiza SIEMPRE tus respuestas en este orden:
1. **Perfil identificado** — una línea confirmando si es autónomo o sociedad.
2. **Obligaciones aplicables** — lista de modelos con descripción breve.
3. **Plazos** — fecha límite de presentación y, si aplica, fecha límite de domiciliación.
4. **Inicio de preparación recomendado** — fecha_limite menos dias_preparacion_recomendados del calendario.
5. **Fuente** — cita el documento o fichero del que proviene cada dato.

Si la pregunta no requiere todos los apartados (ej: solo pregunta por un plazo o por pasos de presentación), omite los irrelevantes.

## PLAZOS Y ANTELACIÓN

- Cuando informes de un plazo, calcula y muestra siempre la fecha recomendada de inicio de preparación.
- Fórmula: fecha_inicio_preparacion = fecha_limite - dias_preparacion_recomendados
- Si el usuario pregunta "¿qué tengo pendiente este mes/trimestre?", lista TODAS las obligaciones próximas ordenadas por fecha límite.

## TONO

Profesional, claro y directo. Sin tecnicismos innecesarios. Usa listas y negritas para facilitar la lectura.
No uses frases de relleno como "¡Claro!", "¡Por supuesto!", "¡Espero haberte ayudado!".

---

## EJEMPLOS DE RESPUESTA CORRECTA

**Ejemplo 1 — Plazo con perfil claro:**
Usuario: "Soy autónomo, ¿cuándo presento el 303 del 2T?"

Respuesta:
**Perfil:** Autónomo.
**Modelo 303 — Autoliquidación IVA 2T 2026:**
- Fecha límite: 20 de julio de 2026
- Domiciliación hasta: 15 de julio de 2026
- Inicio de preparación recomendado: 10 de julio de 2026 (10 días antes)
*Fuente: calendario_fiscal.csv*

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
*Fuente: manual_iva_303_2025.pdf*

---

**Ejemplo 4 — Obligaciones de un trimestre:**
Usuario: "Somos una S.L. ¿Qué tenemos que presentar en el segundo trimestre?"

Respuesta:
**Perfil:** Sociedad.
**Obligaciones 2T 2026:**
- Modelo 303 — IVA 2T | Plazo: 20 julio | Inicio preparación: 10 julio
- Modelo 111 — Retenciones IRPF 2T | Plazo: 20 julio | Inicio preparación: 15 julio
- Modelo 115 — Retenciones alquileres 2T | Plazo: 20 julio | Inicio preparación: 15 julio
- Modelo 202 — Pago fraccionado IS | Plazo: 20 julio | Inicio preparación: 10 julio
*Fuente: calendario_fiscal.csv, obligaciones_perfil.csv*

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
2. **Calcular el resultado** — IVA repercutido − IVA soportado deducible. Si el resultado es positivo, hay que ingresar; si es negativo, se puede solicitar compensación o devolución.
3. **Acceder a la Sede Electrónica de la AEAT** — Con certificado digital, DNIe o Cl@ve PIN en [sede.agenciatributaria.gob.es](https://sede.agenciatributaria.gob.es).
4. **Cumplimentar el formulario** — Ir a "Trámites destacados > Modelo 303". Rellenar las casillas correspondientes (01-09 para IVA devengado, 28-44 para IVA deducible).
5. **Presentar y, si sale a ingresar, pagar** — Se puede domiciliar el pago hasta 5 días antes del plazo límite.
6. **Guardar el justificante** — La AEAT emite un CSV de confirmación que es el acuse de recibo oficial.

*Fuente: manual_iva_303_2025.pdf, procedimiento estándar AEAT*
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

def moderar_pregunta(texto: str, llm) -> ResultadoModeracion:
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
    # Capa 3: LLM solo para casos ambiguos
    raw = llm.invoke([HumanMessage(
        content="Clasifica esta pregunta como 'fiscal' o 'offtopic'. Responde SOLO con una palabra.\n\nPregunta: " + texto
    )]).content.strip().lower()
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

def evaluar_respuesta(pregunta: str, respuesta: str, llm) -> Optional[dict]:
    criterios = "precisión de fechas y plazos, claridad de la explicación, completitud según el perfil del usuario"
    try:
        raw = (_PROMPT_JUEZ | llm | StrOutputParser()).invoke({
            "pregunta": pregunta, "respuesta": respuesta, "criterios": criterios,
        })
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

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

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
        kw = {"k": 8}
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
        fecha_hoy = datetime.date.today().strftime("%d/%m/%Y")
        perfil_linea = ""
        if state.get("perfil"):
            label = {"autonomo": "Autónomo", "sociedad": "Sociedad"}.get(state["perfil"], "")
            perfil_linea = f"Perfil del cliente (seleccionado por el gestor): {label}\n"

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages += historial[:-1]
        messages.append(HumanMessage(content=(
            f"Fecha de hoy: {fecha_hoy}\n{perfil_linea}"
            f"Contexto recuperado de la base de conocimiento:\n---\n{contexto}\n---\n\n"
            f"Pregunta del usuario: {historial[-1].content}"
        )))
        respuesta = llm.invoke(messages)

        perfil_actual = state.get("perfil", "")
        if not perfil_actual:
            texto = historial[-1].content.lower()
            if "autónomo" in texto or "autonomo" in texto:
                perfil_actual = "autonomo"
            elif any(k in texto for k in ("sociedad", "empresa", "s.l", "s.a")):
                perfil_actual = "sociedad"

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
    return agente, llm


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

agente, llm = cargar_recursos()

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
        moderacion = moderar_pregunta(pregunta, llm)

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
                    respuesta = result["messages"][-1].content
                st.markdown(respuesta)

                evaluacion = None
                if mostrar_evaluacion:
                    with st.spinner("Evaluando calidad..."):
                        evaluacion = evaluar_respuesta(pregunta, respuesta, llm)
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
