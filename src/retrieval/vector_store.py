import uuid
from typing import List, Dict, Any
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

from src.retrieval.client import qdrant_client
from src.telemetry import get_tracer

tracer = get_tracer()


class QdrantVectorStore:
    def __init__(self, collection_name: str = "child_chunks"):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert_chunks(self, parents: List[Dict[str, Any]], children: List[Dict[str, Any]]) -> None:
        """Embeds child chunks and upserts them into Qdrant alongside parent payload context."""
        with tracer.start_as_current_span("vector_db.upsert_chunks") as span:
            span.set_attribute("vector_db.parent_count", len(parents))
            span.set_attribute("vector_db.child_count", len(children))

            if not children:
                return

            parent_lookup = {p["parent_id"]: p["text"] for p in parents}
            child_texts = [c["text"] for c in children]
            embeddings = list(self.embedding_model.embed(child_texts))

            points = []
            for child, idx_vector in zip(children, embeddings):
                parent_text = parent_lookup.get(child["parent_id"], "")
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, child["child_id"]))

                point = PointStruct(
                    id=point_id,
                    vector=idx_vector.tolist(),
                    payload={
                        "child_id": child["child_id"],
                        "parent_id": child["parent_id"],
                        "doc_id": child.get("doc_id", ""),
                        "text": parent_text,
                        "child_text": child["text"],
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """Searches top_k child vectors matching query and returns candidates for reranking."""
        with tracer.start_as_current_span("vector_db.search_child_vectors") as span:
            span.set_attribute("db.query", query)
            span.set_attribute("db.top_k", top_k)

            query_vector = list(self.embedding_model.embed([query]))[0].tolist()

            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k
                )
                search_results = response.points
            else:
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )

            candidates = []
            for hit in search_results:
                payload = getattr(hit, "payload", {}) or {}
                candidates.append({
                    "child_id": payload.get("child_id", ""),
                    "parent_id": payload.get("parent_id", ""),
                    "text": payload.get("text", ""),
                    "child_text": payload.get("child_text", ""),
                    "score": float(hit.score)
                })

            if candidates:
                span.set_attribute("retrieval.top_vector_score", candidates[0]["score"])

            return candidates