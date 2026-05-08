import operator
from dataclasses import dataclass
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


# ── Estado del agente LangGraph ────────────────────────────────────────────────
# Annotated[list, operator.add] permite que cada nodo añada mensajes al historial
# sin sobrescribir los anteriores (reducer de LangGraph).
class AgentState(TypedDict):
    messages:      Annotated[list, operator.add]
    perfil:        str   # "autonomo" | "sociedad" | ""
    contexto_rag:  str   # fragmentos recuperados de ChromaDB
    tipo_consulta: str   # "plazos" | "documentos" | "general"


# ── Resultado de moderación ────────────────────────────────────────────────────
# Encapsula la decisión de la capa de moderación en cascada
# (reglas → ML → LLM) para saber si la pregunta es fiscal o fuera de ámbito.
@dataclass
class ResultadoModeracion:
    decision:  str    # "fiscal" | "offtopic"
    capa:      str    # "reglas" | "ml" | "llm"
    confianza: float
