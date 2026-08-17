from typing import List, Dict, Any, Tuple
from src.telemetry import get_tracer

tracer = get_tracer()


class ParentChildChunker:
    def __init__(self, parent_size: int = 1024, child_size: int = 256):
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk(self, text: str, doc_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Splits clean text into 1024-character Parent chunks and 256-character Child chunks."""
        with tracer.start_as_current_span("ingestion.parent_child_chunking") as span:
            parents = []
            children = []

            if not text:
                span.set_attribute("chunking.parent_count", 0)
                span.set_attribute("chunking.child_count", 0)
                return parents, children

            parent_idx = 0
            for i in range(0, len(text), self.parent_size):
                parent_text = text[i : i + self.parent_size]
                parent_id = f"{doc_id}_P{parent_idx}"
                parent_idx += 1

                parents.append({
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "text": parent_text
                })

                child_idx = 0
                for j in range(0, len(parent_text), self.child_size):
                    child_text = parent_text[j : j + self.child_size]
                    child_id = f"{parent_id}_C{child_idx}"
                    child_idx += 1

                    children.append({
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "text": child_text
                    })

            span.set_attribute("chunking.parent_count", len(parents))
            span.set_attribute("chunking.child_count", len(children))
            return parents, children

    # Method aliases for flexibility
    def split(self, text: str, doc_id: str):
        return self.chunk(text, doc_id)

    def process(self, text: str, doc_id: str):
        return self.chunk(text, doc_id)