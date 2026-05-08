import datetime
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app_refactor.core.config import MAX_MESSAGES, SYSTEM_PROMPT
from app_refactor.models.schemas import AgentState
from app_refactor.services.llm_service import _extraer_texto, invoke_con_retry
from app_refactor.services.rag_service import (
    clasificar_consulta,
    recuperar_documentos,
    recuperar_general,
    recuperar_plazos,
)


# ── Nodo: poda de historial ────────────────────────────────────────────────────
# Elimina los mensajes más antiguos cuando el historial supera MAX_MESSAGES
# para evitar que el contexto crezca indefinidamente y agote la ventana del LLM.
def podar_historial(state: AgentState) -> AgentState:
    mensajes = state["messages"]
    if len(mensajes) <= MAX_MESSAGES:
        return {}
    n = len(mensajes) - MAX_MESSAGES
    if n % 2 != 0:
        n += 1  # eliminamos siempre pares user/assistant para no romper el historial
    return {"messages": [RemoveMessage(id=m.id) for m in mensajes[:n]]}


# ── Nodo: detección de perfil ──────────────────────────────────────────────────
# Extrae el perfil (autónomo/sociedad) del historial si no se ha detectado aún.
# El perfil persiste en el estado durante toda la conversación.
_KW_AUTO = re.compile(r"\baut[oó]nomo\b", re.IGNORECASE)
_KW_SOC  = re.compile(r"\b(sociedad|empresa|s\.l|s\.a)\b", re.IGNORECASE)

def detectar_perfil(state: AgentState) -> AgentState:
    if state.get("perfil"):
        return {}
    for msg in reversed(state["messages"]):
        texto = msg.content
        if _KW_AUTO.search(texto):
            return {"perfil": "autonomo"}
        if _KW_SOC.search(texto):
            return {"perfil": "sociedad"}
    return {}


# ── Router: decide qué nodo RAG ejecutar ──────────────────────────────────────
def router(state: AgentState) -> Literal["recuperar_plazos", "recuperar_documentos", "recuperar_general"]:
    return {
        "plazos":     "recuperar_plazos",
        "documentos": "recuperar_documentos",
        "general":    "recuperar_general",
    }[state["tipo_consulta"]]


# ── Nodo: generación de respuesta ──────────────────────────────────────────────
# Construye el mensaje final combinando: system prompt + historial + contexto RAG
# + fecha/trimestre actuales + perfil del cliente.
def generar_respuesta(state: AgentState, llm) -> AgentState:
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

    respuesta = invoke_con_retry(llm, messages, tipo="agente")
    return {"messages": [AIMessage(content=_extraer_texto(respuesta))], "perfil": perfil_actual}


# ── Construcción del grafo ─────────────────────────────────────────────────────
def construir_grafo(llm):
    """Recibe el LLM inicializado y devuelve el grafo compilado con MemorySaver."""

    # Wrapper para inyectar el LLM en el nodo de generación
    def _generar(state: AgentState) -> AgentState:
        return generar_respuesta(state, llm)

    workflow = StateGraph(AgentState)

    workflow.add_node("podar_historial",      podar_historial)
    workflow.add_node("detectar_perfil",      detectar_perfil)
    workflow.add_node("clasificar_consulta",  clasificar_consulta)
    workflow.add_node("recuperar_plazos",     recuperar_plazos)
    workflow.add_node("recuperar_documentos", recuperar_documentos)
    workflow.add_node("recuperar_general",    recuperar_general)
    workflow.add_node("generar_respuesta",    _generar)

    workflow.add_edge(START,                  "podar_historial")
    workflow.add_edge("podar_historial",      "detectar_perfil")
    workflow.add_edge("detectar_perfil",      "clasificar_consulta")
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

    return workflow.compile(checkpointer=MemorySaver())
