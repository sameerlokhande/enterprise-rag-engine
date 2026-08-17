from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
from typing import List, Dict
from src.telemetry import get_tracer

tracer = get_tracer()

class QdrantVectorStore:
    def __init__(self, collection_name: str = "enterprise_docs"):
        self.client = QdrantClient(path="./qdrant_db")
        self.collection_name = collection_name
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def index_chunks(self, chunks: Dict[str, List[Dict]]) -> int:
        with tracer.start_as_current_span("vector_db.index_chunks") as span:
            parents_map = {p["parent_id"]: p["text"] for p in chunks["parents"]}
            children = chunks["children"]
            if not children:
                span.set_attribute("indexed.count", 0)
                return 0

            child_texts = [c["text"] for c in children]
            embeddings = list(self.embed_model.embed(child_texts))

            points = []
            for idx, child in enumerate(children):
                parent_text = parents_map.get(child["parent_id"], child["text"])
                point = PointStruct(
                    id=abs(hash(child["child_id"])) % (10**8),
                    vector=embeddings[idx].tolist(),
                    payload={
                        "child_id": child["child_id"],
                        "parent_id": child["parent_id"],
                        "doc_id": child["doc_id"],
                        "child_text": child["text"],
                        "parent_text": parent_text
                    }
                )
                points.append(point)

            self.client.upsert(collection_name=self.collection_name, points=points)
            span.set_attribute("indexed.points_count", len(points))
            return len(points)

    def search_child_vectors(self, query: str, top_k: int = 30) -> List[Dict]:
        with tracer.start_as_current_span("vector_db.search_child_vectors") as span:
            span.set_attribute("db.query", query)
            span.set_attribute("db.top_k", top_k)

            query_vector = list(self.embed_model.embed([query]))[0].tolist()
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k
            ).points

            candidates = []
            for res in results:
                candidates.append({
                    "child_id": res.payload["child_id"],
                    "parent_id": res.payload["parent_id"],
                    "text": res.payload["parent_text"],
                    "child_text": res.payload["child_text"],
                    "vector_score": float(res.score)
                })

            span.set_attribute("retrieval.candidates_returned", len(candidates))
            if candidates:
                span.set_attribute("retrieval.top_vector_score", candidates[0]["vector_score"])

            return candidates