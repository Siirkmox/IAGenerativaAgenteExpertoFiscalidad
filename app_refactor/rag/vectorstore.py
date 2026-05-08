from pathlib import Path

import pandas as pd
import pdfplumber
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_refactor.core.config import (
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    MANUAL_METADATA,
)
from app_refactor.rag.embeddings import embeddings
from app_refactor.utils.logger import logger


# ── Carga de PDFs ──────────────────────────────────────────────────────────────
# Usa chunking semántico como estrategia principal. Si falla (p.ej. PDF muy corto
# o sin texto suficiente), cae al splitter por caracteres como fallback.
def _cargar_pdfs(metadata_map: dict) -> list[Document]:
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

    docs = []
    for (directorio, filename), meta in metadata_map.items():
        path = directorio / filename
        if not path.exists():
            logger.warning(f"PDF no encontrado: {path}")
            continue
        textos = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    textos.append(t)
        doc_base = Document(
            page_content="\n\n".join(textos),
            metadata={**meta, "fuente": filename},
        )
        try:
            chunks = semantic_splitter.split_documents([doc_base])
            if not chunks:
                raise ValueError("0 chunks semánticos")
        except Exception:
            chunks = fallback_splitter.split_documents([doc_base])
        docs.extend(chunks)
        logger.info(f"PDF indexado: {filename} → {len(chunks)} chunks")
    return docs


# ── Carga de CSVs ──────────────────────────────────────────────────────────────
# Cada fila del CSV es un chunk independiente: cada obligación fiscal es
# una unidad semántica completa (modelo + fecha + perfil + trimestre).
def _cargar_csv(path: Path, tipo: str, sep: str = ",") -> list[Document]:
    df = pd.read_csv(path, sep=sep)
    docs = []
    for _, row in df.iterrows():
        contenido = " | ".join(
            f"{col}: {val}" for col, val in row.items() if pd.notna(val)
        )
        meta = {"fuente": path.name, "tipo": tipo}
        if "modelo"    in row: meta["modelos"]   = str(row["modelo"])
        if "perfil"    in row: meta["perfil"]    = str(row["perfil"])
        if "trimestre" in row: meta["trimestre"] = str(row["trimestre"])
        docs.append(Document(page_content=contenido, metadata=meta))
    logger.info(f"CSV indexado: {path.name} → {len(docs)} filas")
    return docs


# ── Inicialización del vectorstore ─────────────────────────────────────────────
# Si ya existe chroma_db en disco, lo carga directamente (evita re-indexar).
# Si no existe, indexa todos los documentos y persiste la colección.
def init_vectorstore() -> Chroma:
    if Path(CHROMA_DIR).exists():
        logger.info("Cargando ChromaDB existente desde disco.")
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )

    logger.info("Primera ejecución: indexando todos los documentos...")
    all_docs = (
        _cargar_pdfs(MANUAL_METADATA)
        + _cargar_csv(BASE_DIR / "data/calendario_fiscal.csv",   "calendario",          sep=",")
        + _cargar_csv(BASE_DIR / "data/obligaciones_perfil.csv", "obligaciones_perfil", sep=";")
    )
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    logger.info(f"ChromaDB creado con {len(all_docs)} chunks.")
    return vectorstore


vectorstore = init_vectorstore()
retriever   = vectorstore.as_retriever()
