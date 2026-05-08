import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import (
    GOOGLE_API_KEYS,
    MAX_RETRIES_RPM,
    MODELOS_AGENTE,
    MODELOS_LITE,
)
from utils.logger import logger


# ── Estado global de fallback ──────────────────────────────────────────────────
# Rastrea qué combinación modelo+clave está activa para cada tipo de LLM.
# El agente usa modelos potentes primero; lite/judge usa los ligeros primero.
_estado = {
    "agente": {"modelo_idx": 0, "clave_idx": 0},
    "lite":   {"modelo_idx": 0, "clave_idx": 0},
}


def _crear_llm(modelo: str, clave_idx: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=modelo,
        google_api_key=GOOGLE_API_KEYS[clave_idx],
        temperature=0,
    )


def _extraer_texto(msg) -> str:
    """Extrae texto de un mensaje cuyo .content puede ser str o list (Gemini 2.5+)."""
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


def invoke_con_retry(llm_obj, messages: list, tipo: str = "agente"):
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
                logger.warning(f"RPM alcanzado, esperando {delay}s...")
                time.sleep(delay)
                continue
            if _es_limite_diario(err):
                if not _siguiente_combinacion(tipo):
                    raise RuntimeError("Todas las combinaciones modelo+clave están agotadas.") from e
                logger.warning(f"Límite diario → cambiando a modelo/clave siguiente")
                llm_obj = _llm_actual(tipo)
                continue
            raise
    raise RuntimeError("Se agotaron los reintentos de la API.")


def inicializar_llm(tipo: str = "agente") -> ChatGoogleGenerativeAI:
    """Devuelve el LLM inicial sin hacer ninguna llamada de prueba.
    El fallback actúa en el primer uso real si la combinación está agotada."""
    if not GOOGLE_API_KEYS:
        raise RuntimeError("No hay ninguna GOOGLE_API_KEY configurada.")
    modelos = MODELOS_AGENTE if tipo == "agente" else MODELOS_LITE
    _estado[tipo]["modelo_idx"] = 0
    _estado[tipo]["clave_idx"]  = 0
    return _crear_llm(modelos[0], 0)
