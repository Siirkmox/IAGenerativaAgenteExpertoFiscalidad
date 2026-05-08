import datetime
import sys
import uuid
from pathlib import Path

# Añade app_refactor/ al path para que los imports planos funcionen
# tanto en local (ejecutado desde la raíz) como en Streamlit Cloud
# (ejecutado desde app_refactor/).
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

from services.agent_service import (
    evaluar_respuesta,
    moderar_pregunta,
    run_agent,
)
from services.llm_service import _extraer_texto

# BASE_DIR apunta a la raíz del proyecto para leer el CSV del calendario
BASE_DIR = Path(__file__).parent.parent

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asistente Fiscal — GestorIA",
    page_icon="📋",
    layout="centered",
)

st.title("📋 Asistente Fiscal GestorIA")
st.caption("Asesor fiscal para autónomos y sociedades en España · Powered by Gemini + RAG")

tab_chat, tab_calendario = st.tabs(["💬 Asistente", "📅 Calendario Fiscal"])

# ── Sidebar ────────────────────────────────────────────────────────────────────
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

# ── Inicializar estado de sesión ───────────────────────────────────────────────
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

# ── Tab: Chat ──────────────────────────────────────────────────────────────────
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
        moderacion = moderar_pregunta(pregunta)

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
                    state = st.session_state.agent_state
                    state["messages"] = state.get("messages", []) + [HumanMessage(content=pregunta)]
                    result = run_agent(state, st.session_state.thread_id)
                    st.session_state.agent_state = result
                    respuesta = _extraer_texto(result["messages"][-1])
                st.markdown(respuesta)

                evaluacion = None
                if mostrar_evaluacion:
                    with st.spinner("Evaluando calidad..."):
                        evaluacion = evaluar_respuesta(pregunta, respuesta)
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

# ── Tab: Calendario Fiscal ─────────────────────────────────────────────────────
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
