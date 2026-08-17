from flashrank import Ranker, RerankRequest
from typing import List, Dict
from src.telemetry import get_tracer

tracer = get_tracer()

class Sub20msReranker:
    def __init__(self):
        # FlashRank automatically downloads "ms-marco-MiniLM-L-12-v2" on first execution
        self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
        with tracer.start_as_current_span("reranker.flashrank") as span:
            span.set_attribute("rerank.input_candidates", len(candidates))
            span.set_attribute("rerank.requested_top_k", top_k)

            if not candidates:
                span.set_attribute("rerank.output_count", 0)
                return []

            passages = [{"id": c["child_id"], "text": c["text"], "parent_id": c["parent_id"]} for c in candidates]
            rerank_req = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_req)

            ranked_chunks = []
            for res in results[:top_k]:
                ranked_chunks.append({
                    "child_id": res["id"],
                    "text": res["text"],
                    "parent_id": res["parent_id"],
                    "relevance_score": float(res["score"])
                })

            span.set_attribute("rerank.output_count", len(ranked_chunks))
            if ranked_chunks:
                span.set_attribute("rerank.top_score", ranked_chunks[0]["relevance_score"])

            return ranked_chunks