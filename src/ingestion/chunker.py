from typing import List, Dict
from src.telemetry import get_tracer

tracer = get_tracer()

class ParentChildChunker:
    def __init__(self, parent_size: int = 1024, child_size: int = 256, overlap: int = 32):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def chunk_document(self, text: str, doc_id: str) -> Dict[str, List[Dict]]:
        with tracer.start_as_current_span("ingestion.parent_child_chunking") as span:
            parents = []
            children = []

            p_step = self.parent_size - self.overlap
            for p_idx, i in enumerate(range(0, len(text), p_step)):
                p_text = text[i:i + self.parent_size]
                parent_id = f"{doc_id}_P_{p_idx}"
                parents.append({"parent_id": parent_id, "doc_id": doc_id, "text": p_text})

                c_step = self.child_size - self.overlap
                for c_idx, j in enumerate(range(0, len(p_text), c_step)):
                    c_text = p_text[j:j + self.child_size]
                    children.append({
                        "child_id": f"{parent_id}_C_{c_idx}",
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "text": c_text
                    })

            span.set_attribute("chunking.parent_count", len(parents))
            span.set_attribute("chunking.child_count", len(children))
            return {"parents": parents, "children": children}