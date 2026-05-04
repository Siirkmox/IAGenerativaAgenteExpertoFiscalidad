import datetime
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing import TypedDict, Annotated
import operator

# ── Configuración ──────────────────────────────────────────────────────────────

load_dotenv()
# En Streamlit Cloud la API key viene de st.secrets; en local del .env
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
CHROMA_DIR      = "chroma_db"
COLLECTION_NAME = "base_fiscal"

# Rutas de documentos (relativas al directorio donde corre app.py)
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
LEYES_METADATA = {
    (LEYES_DIR, "LeyIVA.pdf"):  {"modelos": "303",     "perfil": "ambos",    "tipo": "ley", "idioma": "es"},
    (LEYES_DIR, "LeyIRPF.pdf"): {"modelos": "100,130", "perfil": "autonomo", "tipo": "ley", "idioma": "es"},
    (LEYES_DIR, "LeyIS.pdf"):   {"modelos": "200,202", "perfil": "sociedad", "tipo": "ley", "idioma": "es"},
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

**Ejemplo 1 — Pregunta con perfil claro:**
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

**Ejemplo 3 — Información no disponible:**
Usuario: "¿Cuál es el tipo de recargo de equivalencia para el aceite de oliva?"

Respuesta:
No dispongo de información suficiente sobre este punto en mi base de conocimiento. Te recomiendo consultar la sede electrónica de la AEAT (sede.agenciatributaria.gob.es) o al gestor responsable.
"""

# ── Carga del vectorstore y agente (cacheados para no recargar en cada interacción) ──

def _indexar_documentos(embeddings):
    """Carga PDFs y CSVs, genera embeddings y persiste en ChromaDB. Solo se ejecuta una vez."""
    import pdfplumber
    import pandas as pd
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    def cargar_pdfs(metadata_map):
        docs = []
        for (directorio, filename), meta in metadata_map.items():
            path = directorio / filename
            if not path.exists():
                continue
            texto_completo = []
            with pdfplumber.open(str(path)) as pdf:
                for page in pdf.pages:
                    texto = page.extract_text()
                    if texto:
                        texto_completo.append(texto)
            texto_unido = "\n\n".join(texto_completo)
            doc_base = Document(page_content=texto_unido, metadata={**meta, "fuente": filename})
            docs.extend(splitter.split_documents([doc_base]))
        return docs

    def cargar_csv(path, tipo, sep=","):
        df = pd.read_csv(path, sep=sep)
        docs = []
        for _, row in df.iterrows():
            contenido = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            meta = {"fuente": path.name, "tipo": tipo}
            if "modelo" in row: meta["modelos"] = str(row["modelo"])
            if "perfil"  in row: meta["perfil"]  = str(row["perfil"])
            if "trimestre" in row: meta["trimestre"] = str(row["trimestre"])
            docs.append(Document(page_content=contenido, metadata=meta))
        return docs

    # En Streamlit Cloud (1GB RAM) excluimos las leyes completas para no agotar memoria.
    # Los manuales prácticos + calendario + obligaciones cubren todos los casos de uso.
    all_docs = (
        cargar_pdfs(MANUAL_METADATA) +
        cargar_csv(BASE_DIR / "data/calendario_fiscal.csv",   "calendario",         sep=",") +
        cargar_csv(BASE_DIR / "data/obligaciones_perfil.csv", "obligaciones_perfil", sep=";")
    )
    return Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
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
            collection_name=COLLECTION_NAME
        )
    else:
        # Primera vez en Streamlit Cloud: indexar desde los PDFs del repositorio
        with st.spinner("Primera ejecución: indexando documentos (puede tardar unos minutos)..."):
            vectorstore = _indexar_documentos(embeddings)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        perfil: str
        contexto_rag: str

    def recuperar_contexto(state: AgentState) -> AgentState:
        ultima_pregunta = state["messages"][-1].content
        perfil = state.get("perfil", "")

        search_kwargs_manuales = {"k": 5}
        if perfil in ("autonomo", "sociedad"):
            search_kwargs_manuales["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
        retriever_manuales = vectorstore.as_retriever(search_kwargs=search_kwargs_manuales)
        docs_manuales_res = retriever_manuales.invoke(ultima_pregunta)

        filter_csv = {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}}
        retriever_csv = vectorstore.as_retriever(search_kwargs={"k": 6, "filter": filter_csv})
        docs_csv_res = retriever_csv.invoke(ultima_pregunta)

        vistos = set()
        docs_combinados = []
        for doc in docs_manuales_res + docs_csv_res:
            clave = doc.page_content[:100]
            if clave not in vistos:
                vistos.add(clave)
                docs_combinados.append(doc)

        contexto = "\n\n".join(
            f"[{d.metadata.get('fuente', '?')}]\n{d.page_content}"
            for d in docs_combinados
        )
        return {"contexto_rag": contexto}

    def generar_respuesta(state: AgentState) -> AgentState:
        contexto = state.get("contexto_rag", "")
        historial = state["messages"]

        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages += historial[:-1]

        ultima = historial[-1].content
        fecha_hoy = datetime.date.today().strftime("%d/%m/%Y")
        perfil_linea = ""
        if state.get("perfil"):
            perfil_label = {"autonomo": "Autónomo", "sociedad": "Sociedad"}.get(state["perfil"], "")
            perfil_linea = f"Perfil del cliente (seleccionado por el gestor): {perfil_label}\n"

        prompt_con_contexto = f"""Fecha de hoy: {fecha_hoy}
{perfil_linea}
Contexto recuperado de la base de conocimiento:
---
{contexto}
---

Pregunta del usuario: {ultima}"""

        messages.append(HumanMessage(content=prompt_con_contexto))
        respuesta = llm.invoke(messages)

        perfil_actual = state.get("perfil", "")
        texto = ultima.lower()
        if not perfil_actual:
            if "autónomo" in texto or "autonomo" in texto:
                perfil_actual = "autonomo"
            elif "sociedad" in texto or "empresa" in texto or "s.l" in texto:
                perfil_actual = "sociedad"

        return {
            "messages": [AIMessage(content=respuesta.content)],
            "perfil": perfil_actual
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("recuperar_contexto", recuperar_contexto)
    workflow.add_node("generar_respuesta", generar_respuesta)
    workflow.set_entry_point("recuperar_contexto")
    workflow.add_edge("recuperar_contexto", "generar_respuesta")
    workflow.add_edge("generar_respuesta", END)

    memory = MemorySaver()
    agente = workflow.compile(checkpointer=memory)
    return agente

# ── Interfaz Streamlit ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Asistente Fiscal — GestorIA",
    page_icon="📋",
    layout="centered"
)

st.title("📋 Asistente Fiscal GestorIA")
st.caption("Asesor fiscal para autónomos y sociedades en España · Powered by Gemini + RAG")

tab_chat, tab_calendario = st.tabs(["💬 Asistente", "📅 Calendario Fiscal"])

# Selector de perfil en la barra lateral
with st.sidebar:
    st.header("Configuración")
    perfil_opcion = st.selectbox(
        "Perfil del cliente",
        options=["Sin especificar", "Autónomo", "Sociedad"],
        index=0
    )
    perfil_map = {"Sin especificar": "", "Autónomo": "autonomo", "Sociedad": "sociedad"}
    perfil_seleccionado = perfil_map[perfil_opcion]

    if st.button("🔄 Nueva conversación"):
        st.session_state.messages = []
        st.session_state.agent_state = {
            "messages": [], "perfil": perfil_seleccionado, "contexto_rag": ""
        }
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("**Modelos cubiertos:**")
    st.markdown("303 · 130 · 111 · 115 · 100 · 200 · 202 · 390 · 347")

# Inicializar estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [], "perfil": perfil_seleccionado, "contexto_rag": ""
    }

# Actualizar perfil si cambia el selector
if st.session_state.agent_state.get("perfil") != perfil_seleccionado and not st.session_state.messages:
    st.session_state.agent_state["perfil"] = perfil_seleccionado

# Cargar el agente
agente = cargar_recursos()

with tab_chat:
    # Mostrar historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario — formulario estable compatible con Python 3.14
    with st.form(key="chat_form", clear_on_submit=True):
        pregunta = st.text_input("Escribe tu pregunta fiscal...", key="input_pregunta")
        enviar = st.form_submit_button("Enviar")

    if enviar and pregunta.strip():
        st.session_state.messages.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Consultando la base de conocimiento..."):
                state = st.session_state.agent_state
                state["messages"] = state.get("messages", []) + [HumanMessage(content=pregunta)]

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = agente.invoke(state, config=config)

                st.session_state.agent_state = result
                respuesta = result["messages"][-1].content

            st.markdown(respuesta)

        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        st.rerun()

with tab_calendario:
    st.subheader("Calendario de obligaciones fiscales 2026")

    @st.cache_data
    def cargar_calendario():
        df = pd.read_csv(BASE_DIR / "data/calendario_fiscal.csv")
        df["fecha_limite_2026"] = pd.to_datetime(df["fecha_limite_2026"])
        df["domiciliacion_hasta"] = pd.to_datetime(df["domiciliacion_hasta"], errors="coerce")
        return df.sort_values("fecha_limite_2026")

    df_cal = cargar_calendario()

    # Filtro de perfil
    opciones_perfil = {"Todos": None, "Autónomo": "autonomo", "Sociedad": "sociedad"}
    perfil_filtro = st.radio(
        "Mostrar obligaciones de:",
        options=list(opciones_perfil.keys()),
        index={"": 0, "autonomo": 1, "sociedad": 2}.get(perfil_seleccionado, 0),
        horizontal=True
    )
    valor_filtro = opciones_perfil[perfil_filtro]

    if valor_filtro:
        df_vis = df_cal[df_cal["perfil"].isin([valor_filtro, "ambos"])].copy()
    else:
        df_vis = df_cal.copy()

    # Destacar vencimientos próximos (≤ 30 días desde hoy)
    hoy = datetime.date.today()
    df_vis["_dias_restantes"] = (df_vis["fecha_limite_2026"].dt.date - hoy).apply(lambda d: d.days)
    df_vis = df_vis.reset_index(drop=True)

    def estado(d):
        if d < 0:
            return "✅ Vencido"
        elif d <= 7:
            return "🔴 ≤ 7 días"
        elif d <= 30:
            return "🟡 ≤ 30 días"
        else:
            return "🟢 Pendiente"

    df_show = pd.DataFrame({
        "Estado": df_vis["_dias_restantes"].apply(estado),
        "Modelo": df_vis["modelo"],
        "Obligación": df_vis["nombre"],
        "Perfil": df_vis["perfil"],
        "Fecha límite": df_vis["fecha_limite_2026"].dt.strftime("%d/%m/%Y"),
        "Domiciliación hasta": df_vis["domiciliacion_hasta"].apply(
            lambda x: x.strftime("%d/%m/%Y") if pd.notna(x) else "—"
        ),
        "Días preparación": df_vis["dias_preparacion_recomendados"],
        "Periodicidad": df_vis["periodicidad"],
    })

    st.dataframe(df_show, use_container_width=True, hide_index=True)
