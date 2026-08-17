from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from fastembed import TextEmbedding
from typing import Optional, Dict
from src.telemetry import get_tracer

tracer = get_tracer()

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92):
        self.client = QdrantClient(path="./qdrant_db")
        self.collection_name = "semantic_cache"
        self.threshold = similarity_threshold
        self.embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._ensure_cache_collection()

    def _ensure_cache_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def lookup(self, query: str) -> Optional[Dict]:
        with tracer.start_as_current_span("semantic_cache.lookup") as span:
            span.set_attribute("cache.query", query)
            span.set_attribute("cache.threshold", self.threshold)

            query_vector = list(self.embed_model.embed([query]))[0].tolist()
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1
            ).points

            if results and results[0].score >= self.threshold:
                score = float(results[0].score)
                span.set_attribute("cache.status", "CACHE_HIT")
                span.set_attribute("cache.similarity_score", score)
                return {
                    "cached_answer": results[0].payload["answer"],
                    "cached_query": results[0].payload["query"],
                    "similarity_score": score
                }

            span.set_attribute("cache.status", "CACHE_MISS")
            if results:
                span.set_attribute("cache.best_miss_score", float(results[0].score))
            return None

    def store(self, query: str, answer: str):
        with tracer.start_as_current_span("semantic_cache.store") as span:
            query_vector = list(self.embed_model.embed([query]))[0].tolist()
            point_id = abs(hash(query)) % (10**8)
            
            point = PointStruct(
                id=point_id,
                vector=query_vector,
                payload={"query": query, "answer": answer}
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            span.set_attribute("cache.stored_point_id", point_id)