import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from src.ingestion.normalizer import DocumentNormalizer
from src.ingestion.chunker import ParentChildChunker
from src.retrieval.vector_store import QdrantVectorStore
from src.security.guardrails import SecurityGuard
from src.telemetry import get_tracer

tracer = get_tracer()
security_guard = SecurityGuard()


class IngestionState(TypedDict):
    file_path: str
    raw_text: str
    clean_text: str
    doc_id: str
    status: str


def extract_text_node(state: IngestionState) -> IngestionState:
    """Extracts raw text and embeds page metadata for interactive citations."""
    with tracer.start_as_current_span("langgraph_node.extract_text") as span:
        file_path = state["file_path"]
        ext = os.path.splitext(file_path)[1].lower()

        span.set_attribute("file.path", file_path)
        span.set_attribute("file.extension", ext)

        raw_text = ""
        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                pages_text = []
                for idx, page in enumerate(reader.pages, 1):
                    extracted = page.extract_text() or ""
                    pages_text.append(f"[Page {idx}]\n{extracted}")
                raw_text = "\n\n".join(pages_text)
            except ImportError:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                raw_text = "\n".join([p.text for p in doc.paragraphs])
            except ImportError:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

        # Sanitize PII before persisting
        sanitized_text = security_guard.sanitize_text(raw_text)

        span.set_attribute("extraction.status", "SUCCESS")
        span.set_attribute("extraction.length", len(sanitized_text))

        return {**state, "raw_text": sanitized_text}


def normalize_node(state: IngestionState) -> IngestionState:
    """Cleans Unicode characters, trims extra spaces, and assigns doc_id."""
    normalizer = DocumentNormalizer()
    normalized_doc = normalizer.normalize(state["raw_text"])

    doc_id = getattr(normalized_doc, "doc_id", getattr(normalized_doc, "id", "DOC_1001"))
    clean_text = getattr(
        normalized_doc, 
        "clean_text", 
        getattr(normalized_doc, "text", getattr(normalized_doc, "normalized_text", state["raw_text"]))
    )

    return {**state, "doc_id": doc_id, "clean_text": clean_text}


def chunk_and_index_node(state: IngestionState) -> IngestionState:
    """Splits normalized text into parent/child chunks and indexes vectors into Qdrant."""
    chunker = ParentChildChunker()
    chunks = chunker.chunk(state["clean_text"], state["doc_id"])

    if isinstance(chunks, tuple) and len(chunks) == 2:
        parents, children = chunks
    else:
        parents = getattr(chunks, "parents", [])
        children = getattr(chunks, "children", [])

    vector_store = QdrantVectorStore()
    vector_store.upsert_chunks(parents, children)
    return {**state, "status": "SUCCESS"}


# Assemble LangGraph Workflow
workflow = StateGraph(IngestionState)

workflow.add_node("extract_text", extract_text_node)
workflow.add_node("normalize", normalize_node)
workflow.add_node("chunk_and_index", chunk_and_index_node)

workflow.set_entry_point("extract_text")
workflow.add_edge("extract_text", "normalize")
workflow.add_edge("normalize", "chunk_and_index")
workflow.add_edge("chunk_and_index", END)

# Export compiled graph
ingestion_pipeline = workflow.compile()