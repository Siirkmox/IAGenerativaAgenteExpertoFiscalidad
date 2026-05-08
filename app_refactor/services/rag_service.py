import re

from app_refactor.models.schemas import AgentState
from app_refactor.rag.vectorstore import vectorstore


# ── Keywords para clasificar el tipo de consulta ───────────────────────────────
_KEYWORDS_PLAZOS = re.compile(
    r"\b(plazo|fecha|cuando|cuándo|vencimiento|trimestre|domicili|"
    r"antelacion|antelación|pendiente)\b",
    re.IGNORECASE,
)
_KEYWORDS_DOCS = re.compile(
    r"\b(casilla|rellenar|cumplimentar|calcul|base imponible|deducci[oó]n|"
    r"como se|cómo se|instrucci[oó]n|apartado|anexo|"
    r"paso|pasos|proceso|procedimiento|"
    r"c[oó]mo presento|c[oó]mo se presenta|c[oó]mo funciona|"
    r"c[oó]mo hago|c[oó]mo debo|qu[eé] pasos|qu[eé] debo hacer)\b",
    re.IGNORECASE,
)


def clasificar_consulta(state: AgentState) -> AgentState:
    """Clasifica la última pregunta del usuario para enrutar al nodo RAG correcto."""
    ultima = state["messages"][-1].content
    if _KEYWORDS_DOCS.search(ultima):
        tipo = "documentos"
    elif _KEYWORDS_PLAZOS.search(ultima):
        tipo = "plazos"
    else:
        tipo = "general"
    return {"tipo_consulta": tipo}


# ── Utilidad: combinar docs sin duplicados ─────────────────────────────────────
def _combinar(docs_a: list, docs_b: list) -> str:
    """Une dos listas de documentos eliminando duplicados por los primeros 100 chars."""
    vistos, resultado = set(), []
    for doc in docs_a + docs_b:
        clave = doc.page_content[:100]
        if clave not in vistos:
            vistos.add(clave)
            resultado.append(doc)
    return "\n\n".join(
        f"[{d.metadata.get('fuente', '?')}]\n{d.page_content}" for d in resultado
    )


# ── Nodos RAG especializados ───────────────────────────────────────────────────
# Cada nodo ajusta el número de chunks y los filtros de metadatos según el tipo
# de consulta para maximizar la relevancia del contexto recuperado.

def recuperar_plazos(state: AgentState) -> AgentState:
    """Prioriza CSV de calendario y obligaciones (k=10) + manuales filtrados por perfil (k=3)."""
    ultima = state["messages"][-1].content
    perfil = state.get("perfil", "")

    docs_csv = vectorstore.as_retriever(search_kwargs={
        "k": 10,
        "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
    }).invoke(ultima)

    kw = {"k": 3}
    if perfil in ("autonomo", "sociedad"):
        kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
    docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)

    return {"contexto_rag": _combinar(docs_csv, docs_m)}


def recuperar_documentos(state: AgentState) -> AgentState:
    """Prioriza manuales PDF (k=12) + algo de CSV para contexto de fechas (k=3)."""
    ultima = state["messages"][-1].content
    perfil = state.get("perfil", "")

    kw = {"k": 12}
    if perfil in ("autonomo", "sociedad"):
        kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
    docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)

    docs_csv = vectorstore.as_retriever(search_kwargs={
        "k": 3,
        "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
    }).invoke(ultima)

    return {"contexto_rag": _combinar(docs_m, docs_csv)}


def recuperar_general(state: AgentState) -> AgentState:
    """Balance entre manuales (k=5) y CSV (k=6) para preguntas mixtas o ambiguas."""
    ultima = state["messages"][-1].content
    perfil = state.get("perfil", "")

    kw = {"k": 5}
    if perfil in ("autonomo", "sociedad"):
        kw["filter"] = {"perfil": {"$in": [perfil, "ambos"]}}
    docs_m = vectorstore.as_retriever(search_kwargs=kw).invoke(ultima)

    docs_csv = vectorstore.as_retriever(search_kwargs={
        "k": 6,
        "filter": {"tipo": {"$in": ["calendario", "obligaciones_perfil"]}},
    }).invoke(ultima)

    return {"contexto_rag": _combinar(docs_m, docs_csv)}
