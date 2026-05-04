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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
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

- Usa ÚNICAMENTE la información que aparece literalmente en el contexto proporcionado.
- Si un dato concreto (fecha, casilla, porcentaje, plazo) no aparece textualmente en el contexto, NO lo inventes ni lo inferias.
- En ese caso responde exactamente: "No dispongo de información suficiente sobre este punto en mi base de conocimiento. Te recomiendo consultar la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) o al gestor responsable."

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

Si la pregunta no requiere todos los apartados (ej: solo pregunta por un plazo), omite los irrelevantes.

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
"""

# ── 1. MODERACIÓN EN CASCADA ───────────────────────────────────────────────────
# Filtra preguntas fuera de ámbito antes de llegar al agente.
# Reglas (gratis) → ML (rápido) → LLM (solo casos ambiguos)

UMBRAL_CONFIANZA_ML = 0.85

KEYWORDS_FISCAL = re.compile(
    r"\b(modelo|irpf|iva|impuesto|declaraci[oó]n|renta|hacienda|aeat|tribut|fiscal|"
    r"autonomo|aut[oó]nomo|sociedad|empresa|s\.l|factura|casilla|plazo|trimestre|"
    r"303|130|111|115|100|200|202|347|390|retenci[oó]n|deducci[oó]n)\b",
    re.IGNORECASE,
)
KEYWORDS_OFFTOPIC = re.compile(
    r"\b(receta|cocina|deporte|f[uú]tbol|pel[ií]cula|m[uú]sica|viaje|hotel|"
    r"tiempo|clima|meteorolog[ií]a|amor|relaci[oó]n|juego|videojuego)\b",
    re.IGNORECASE,
)

@dataclass
class ResultadoModeracion:
    decision: str       # "fiscal" | "offtopic" | "ambiguo"
    capa:     str       # "reglas" | "ml" | "llm"
    confianza: float

@st.cache_resource(show_spinner=False)
def _construir_clasificador_ml():
    """Entrena un clasificador TF-IDF + LogisticRegression con ejemplos mínimos."""
    ejemplos = [
        ("¿Cuándo presento el modelo 303?", "fiscal"),
        ("¿Qué obligaciones tengo como autónomo?", "fiscal"),
        ("¿Cómo se rellena la casilla 01 del IVA?", "fiscal"),
        ("Plazo para presentar el IRPF 2025", "fiscal"),
        ("¿Qué es la domiciliación bancaria en el modelo 130?", "fiscal"),
        ("Retenciones en el modelo 111 del segundo trimestre", "fiscal"),
        ("¿Cuánto tiempo tengo para presentar el IS?", "fiscal"),
        ("Deducciones en el modelo 303", "fiscal"),
        ("¿Cómo me doy de alta como autónomo en hacienda?", "fiscal"),
        ("¿Qué es el pago fraccionado del IRPF?", "fiscal"),
        ("¿Cuál es la mejor receta de paella?", "offtopic"),
        ("¿Quién ganó el partido de ayer?", "offtopic"),
        ("Recomiéndame una película de terror", "offtopic"),
        ("¿Qué tiempo hace en Madrid?", "offtopic"),
        ("¿Cómo se llama el presidente de Francia?", "offtopic"),
        ("Cuéntame un chiste", "offtopic"),
        ("¿Cuál es la capital de Australia?", "offtopic"),
        ("Dame una ruta de senderismo", "offtopic"),
    ]
    X, y = zip(*ejemplos)
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(list(X), list(y))
    return clf

def moderar_pregunta(texto: str, llm) -> ResultadoModeracion:
    """Cascada: Reglas → ML → LLM para clasificar si la pregunta es fiscal."""
    # Capa 1: reglas por keywords
    if KEYWORDS_FISCAL.search(texto):
        return ResultadoModeracion("fiscal", "reglas", 1.0)
    if KEYWORDS_OFFTOPIC.search(texto):
        return ResultadoModeracion("offtopic", "reglas", 1.0)

    # Capa 2: ML
    clf = _construir_clasificador_ml()
    proba = clf.predict_proba([texto])[0]
    clases = clf.classes_
    idx_max = int(np.argmax(proba))
    confianza = float(proba[idx_max])
    if confianza >= UMBRAL_CONFIANZA_ML:
        return ResultadoModeracion(clases[idx_max], "ml", confianza)

    # Capa 3: LLM (solo casos ambiguos)
    prompt = (
        "Clasifica esta pregunta como 'fiscal' o 'offtopic'. "
        "Responde SOLO con una palabra.\n\n"
        f"Pregunta: {texto}"
    )
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    decision = "fiscal" if "fiscal" in raw else "offtopic"
    return ResultadoModeracion(decision, "llm", 0.6)


# ── 2. HERRAMIENTAS FISCALES (@tool) ──────────────────────────────────────────
# El LLM puede invocar estas herramientas dinámicamente según la pregunta.
# Se inicializan con el DataFrame del calendario en tiempo de carga.

def _crear_tools(calendario_path: Path):
    df_cal = pd.read_csv(calendario_path)
    df_cal["fecha_limite_2026"] = pd.to_datetime(df_cal["fecha_limite_2026"]).dt.date

    @tool
    def consultar_obligaciones_proximas(perfil: str, dias_horizonte: int = 60) -> str:
        """Devuelve las obligaciones fiscales próximas para un perfil dado.
        Usar cuando el usuario pregunte qué tiene pendiente en los próximos días o meses.
        perfil: 'autonomo', 'sociedad' o 'ambos'
        dias_horizonte: número de días a mirar hacia adelante (por defecto 60)
        """
        hoy   = datetime.date.today()
        hasta = hoy + datetime.timedelta(days=dias_horizonte)
        df = df_cal.copy()
        if perfil in ("autonomo", "sociedad"):
            df = df[df["perfil"].isin([perfil, "ambos"])]
        df = df[(df["fecha_limite_2026"] >= hoy) & (df["fecha_limite_2026"] <= hasta)]
        df = df.sort_values("fecha_limite_2026")
        if df.empty:
            return f"No hay obligaciones en los próximos {dias_horizonte} días."
        lineas = [f"Obligaciones {hoy} → {hasta}:\n"]
        for _, row in df.iterrows():
            inicio = row["fecha_limite_2026"] - datetime.timedelta(days=int(row["dias_preparacion_recomendados"]))
            lineas.append(
                f"• Modelo {row['modelo']} — {row['nombre']}\n"
                f"  Plazo: {row['fecha_limite_2026']} | Inicio recomendado: {inicio}"
            )
        return "\n".join(lineas)

    @tool
    def consultar_plazo_modelo(modelo: str) -> str:
        """Devuelve el plazo exacto de un modelo fiscal concreto.
        Usar cuando el usuario pregunte por la fecha límite de un modelo específico.
        modelo: número del modelo (ej: '303', '130', '111')
        """
        df = df_cal[df_cal["modelo"].astype(str) == str(modelo)]
        if df.empty:
            return f"No encontré información del modelo {modelo} en el calendario."
        lineas = []
        for _, row in df.iterrows():
            inicio = row["fecha_limite_2026"] - datetime.timedelta(days=int(row["dias_preparacion_recomendados"]))
            lineas.append(
                f"Modelo {row['modelo']} — {row['nombre']} ({row['perfil']})\n"
                f"  Plazo: {row['fecha_limite_2026']} | Inicio recomendado: {inicio}"
            )
        return "\n".join(lineas)

    return [consultar_obligaciones_proximas, consultar_plazo_modelo]


# ── 3. LLM-AS-JUDGE ───────────────────────────────────────────────────────────

PROMPT_JUEZ = ChatPromptTemplate.from_template("""
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
    """Evalúa una respuesta del agente con un segundo LLM como juez."""
    criterios = "precisión de fechas y plazos, claridad de la explicación, completitud según el perfil del usuario"
    try:
        chain = PROMPT_JUEZ | llm | StrOutputParser()
        raw = chain.invoke({"pregunta": pregunta, "respuesta": respuesta, "criterios": criterios})
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception:
        return None


# ── 4. VECTORSTORE + AGENTE (cacheados) ───────────────────────────────────────

def _indexar_documentos(embeddings):
    import pdfplumber
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

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
            docs.extend(splitter.split_documents([doc_base]))
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

    all_docs = (
        cargar_pdfs(MANUAL_METADATA)
        + cargar_csv(BASE_DIR / "data/calendario_fiscal.csv",    "calendario",          sep=",")
        + cargar_csv(BASE_DIR / "data/obligaciones_perfil.csv",  "obligaciones_perfil", sep=";")
    )
    return Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
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
        with st.spinner("Primera ejecución: indexando documentos..."):
            vectorstore = _indexar_documentos(embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    # Herramientas fiscales y LLM con tools enlazadas
    tools = _crear_tools(BASE_DIR / "data/calendario_fiscal.csv")
    llm_con_tools = llm.bind_tools(tools)
    nodo_tools = ToolNode(tools)

    # ── Estado del agente ──
    class AgentState(TypedDict):
        messages:     Annotated[list, operator.add]
        perfil:       str
        contexto_rag: str

    # ── Nodos ──
    def recuperar_contexto(state: AgentState) -> AgentState:
        ultima = state["messages"][-1].content
        perfil = state.get("perfil", "")

        search_manuales = {"k": 5}
        if perfil in ("autonomo", "sociedad"):
            search_manuales["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
        docs_m = vectorstore.as_retriever(search_kwargs=search_manuales).invoke(ultima)

        docs_csv = vectorstore.as_retriever(search_kwargs={
            "k": 6,
            "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
        }).invoke(ultima)

        vistos, combinados = set(), []
        for doc in docs_m + docs_csv:
            clave = doc.page_content[:100]
            if clave not in vistos:
                vistos.add(clave)
                combinados.append(doc)

        contexto = "\n\n".join(
            f"[{d.metadata.get('fuente','?')}]\n{d.page_content}" for d in combinados
        )
        return {"contexto_rag": contexto}

    def nodo_agente(state: AgentState) -> AgentState:
        """LLM con tools. Puede invocar herramientas o responder directamente."""
        contexto = state.get("contexto_rag", "")
        historial = state["messages"]

        fecha_hoy = datetime.date.today().strftime("%d/%m/%Y")
        perfil_linea = ""
        if state.get("perfil"):
            label = {"autonomo": "Autónomo", "sociedad": "Sociedad"}.get(state["perfil"], "")
            perfil_linea = f"Perfil del cliente: {label}\n"

        system_completo = (
            SYSTEM_PROMPT
            + f"\n\nFecha de hoy: {fecha_hoy}\n{perfil_linea}"
            + f"\nContexto RAG:\n---\n{contexto}\n---"
        )

        messages = [SystemMessage(content=system_completo)] + historial
        respuesta = llm_con_tools.invoke(messages)
        return {"messages": [respuesta]}

    def actualizar_perfil(state: AgentState) -> AgentState:
        """Detecta el perfil del cliente si aún no está definido."""
        perfil_actual = state.get("perfil", "")
        if perfil_actual:
            return {}
        ultima = state["messages"][-1].content if state["messages"] else ""
        texto = ultima.lower()
        if "autónomo" in texto or "autonomo" in texto:
            return {"perfil": "autonomo"}
        if any(k in texto for k in ("sociedad", "empresa", "s.l", "s.a")):
            return {"perfil": "sociedad"}
        return {}

    def router_react(state: AgentState) -> Literal["tools", "actualizar_perfil"]:
        """Si el LLM pidió una tool, ejecutarla; si no, actualizar perfil y terminar."""
        ultimo = state["messages"][-1]
        if hasattr(ultimo, "tool_calls") and ultimo.tool_calls:
            return "tools"
        return "actualizar_perfil"

    # ── Grafo ReAct ──
    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar_contexto", recuperar_contexto)
    workflow.add_node("agente",             nodo_agente)
    workflow.add_node("tools",              nodo_tools)
    workflow.add_node("actualizar_perfil",  actualizar_perfil)

    workflow.add_edge(START,               "recuperar_contexto")
    workflow.add_edge("recuperar_contexto", "agente")
    workflow.add_conditional_edges(
        "agente",
        router_react,
        {"tools": "tools", "actualizar_perfil": "actualizar_perfil"},
    )
    workflow.add_edge("tools",             "agente")   # loop ReAct
    workflow.add_edge("actualizar_perfil", END)

    memory = MemorySaver()
    agente = workflow.compile(checkpointer=memory)
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
    perfil_map       = {"Sin especificar": "", "Autónomo": "autonomo", "Sociedad": "sociedad"}
    perfil_seleccionado = perfil_map[perfil_opcion]

    mostrar_evaluacion = st.toggle("Mostrar evaluación de respuestas", value=False)

    if st.button("🔄 Nueva conversación"):
        st.session_state.messages    = []
        st.session_state.agent_state = {"messages": [], "perfil": perfil_seleccionado, "contexto_rag": ""}
        st.session_state.thread_id   = str(uuid.uuid4())
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
    st.session_state.agent_state = {"messages": [], "perfil": perfil_seleccionado, "contexto_rag": ""}

if st.session_state.agent_state.get("perfil") != perfil_seleccionado and not st.session_state.messages:
    st.session_state.agent_state["perfil"] = perfil_seleccionado

agente, llm = cargar_recursos()

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evaluacion") and mostrar_evaluacion:
                ev = msg["evaluacion"]
                with st.expander(f"Evaluación LLM-as-Judge — {ev.get('puntuacion_global','-')}/10"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Precisión", f"{ev.get('precision_tecnica','-')}/10")
                    col2.metric("Claridad",  f"{ev.get('claridad','-')}/10")
                    col3.metric("Completitud", f"{ev.get('completitud','-')}/10")
                    st.caption(ev.get("justificacion", ""))

    pregunta = st.chat_input("Escribe tu pregunta fiscal...")

    if pregunta and pregunta.strip():
        # Moderación en cascada antes de llegar al agente
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

                    # Extraer la última respuesta de texto (ignorar ToolMessages)
                    respuesta = next(
                        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
                        ""
                    )

                st.markdown(respuesta)

                # LLM-as-Judge (solo si el toggle está activo)
                evaluacion = None
                if mostrar_evaluacion:
                    with st.spinner("Evaluando calidad de la respuesta..."):
                        evaluacion = evaluar_respuesta(pregunta, respuesta, llm)
                    if evaluacion:
                        with st.expander(f"Evaluación LLM-as-Judge — {evaluacion.get('puntuacion_global','-')}/10"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Precisión",   f"{evaluacion.get('precision_tecnica','-')}/10")
                            col2.metric("Claridad",    f"{evaluacion.get('claridad','-')}/10")
                            col3.metric("Completitud", f"{evaluacion.get('completitud','-')}/10")
                            st.caption(evaluacion.get("justificacion", ""))

            st.session_state.messages.append({
                "role":       "assistant",
                "content":    respuesta,
                "evaluacion": evaluacion,
            })

        st.rerun()

with tab_calendario:
    st.subheader("Calendario de obligaciones fiscales 2026")

    @st.cache_data
    def cargar_calendario():
        df = pd.read_csv(BASE_DIR / "data/calendario_fiscal.csv")
        df["fecha_limite_2026"]  = pd.to_datetime(df["fecha_limite_2026"])
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

    df_show = pd.DataFrame({
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
    })

    st.dataframe(df_show, use_container_width=True, hide_index=True)
