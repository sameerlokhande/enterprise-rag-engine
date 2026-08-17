import os
import pandas as pd
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup

from src.ingestion.normalizer import DocumentNormalizer
from src.ingestion.chunker import ParentChildChunker
from src.retrieval.vector_store import QdrantVectorStore
from src.telemetry import get_tracer

tracer = get_tracer()

class IngestionState(TypedDict):
    file_path: str
    file_type: str
    raw_text: str
    normalized_doc: Any
    chunks: Dict[str, List[Dict]]
    status: str
    error: str

def route_and_extract_node(state: IngestionState) -> Dict:
    with tracer.start_as_current_span("langgraph_node.extract_text") as span:
        file_path = state["file_path"]
        ext = os.path.splitext(file_path)[1].lower()
        span.set_attribute("file.path", file_path)
        span.set_attribute("file.extension", ext)
        extracted_text = ""

        try:
            if ext == ".pdf":
                reader = PdfReader(file_path)
                extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            elif ext in [".docx", ".doc"]:
                doc = docx.Document(file_path)
                extracted_text = "\n".join([p.text for p in doc.paragraphs])
            elif ext in [".xlsx", ".xls", ".csv"]:
                df = pd.read_csv(file_path) if ext == ".csv" else pd.read_excel(file_path)
                extracted_text = df.to_markdown(index=False)
            elif ext in [".html", ".htm"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = BeautifulSoup(f.read(), "html.parser").get_text(separator="\n")
            elif ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            else:
                span.set_attribute("extraction.status", "UNSUPPORTED")
                return {"status": "FAILED", "error": f"Unsupported file type: {ext}"}

            span.set_attribute("extraction.status", "SUCCESS")
            span.set_attribute("extraction.length", len(extracted_text))
            return {"raw_text": extracted_text, "file_type": ext, "status": "EXTRACTED"}
        except Exception as e:
            span.record_exception(e)
            return {"status": "FAILED", "error": str(e)}

def normalize_node(state: IngestionState) -> Dict:
    if state["status"] == "FAILED": return {}
    normalizer = DocumentNormalizer()
    norm_doc = normalizer.normalize(raw_text=state["raw_text"], source_path=state["file_path"])
    return {"normalized_doc": norm_doc, "status": "NORMALIZED"}

def chunk_and_index_node(state: IngestionState) -> Dict:
    if state["status"] == "FAILED": return {}
    norm_doc = state["normalized_doc"]
    chunker = ParentChildChunker(parent_size=1024, child_size=256)
    chunks = chunker.chunk_document(text=norm_doc.clean_text, doc_id=norm_doc.doc_id)
    
    vector_store = QdrantVectorStore()
    vector_store.index_chunks(chunks)
    return {"chunks": chunks, "status": "SUCCESS"}

def build_ingestion_graph():
    workflow = StateGraph(IngestionState)
    workflow.add_node("extract", route_and_extract_node)
    workflow.add_node("normalize", normalize_node)
    workflow.add_node("chunk_and_index", chunk_and_index_node)

    workflow.set_entry_point("extract")
    workflow.add_edge("extract", "normalize")
    workflow.add_edge("normalize", "chunk_and_index")
    workflow.add_edge("chunk_and_index", END)
    return workflow.compile()