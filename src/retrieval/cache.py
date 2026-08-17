import uuid
from typing import Optional
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding

from src.retrieval.client import qdrant_client
from src.telemetry import get_tracer

tracer = get_tracer()


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, collection_name: str = "semantic_cache"):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.threshold = similarity_threshold
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the cache collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def _query_vectors(self, query_vector: list, limit: int = 1, session_id: Optional[str] = None):
        """Supports query_points and search with optional session payload filtering."""
        query_filter = None
        if session_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="session_id",
                        match=MatchValue(value=session_id)
                    )
                ]
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            return response.points
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit
        )

    def get(self, query: str, session_id: Optional[str] = None) -> Optional[str]:
        """Looks up semantically similar queries matching the current session ID."""
        with tracer.start_as_current_span("semantic_cache.get") as span:
            span.set_attribute("cache.query", query)
            if session_id:
                span.set_attribute("cache.session_id", session_id)

            query_vector = list(self.embedding_model.embed([query]))[0].tolist()
            search_results = self._query_vectors(query_vector, limit=1, session_id=session_id)

            if search_results and search_results[0].score >= self.threshold:
                span.set_attribute("cache.status", "HIT")
                span.set_attribute("cache.similarity_score", float(search_results[0].score))
                payload = getattr(search_results[0], "payload", {}) or {}
                return payload.get("response")

            span.set_attribute("cache.status", "MISS")
            return None

    def set(self, query: str, response: str, session_id: Optional[str] = None) -> None:
        """Stores query embedding and response tagged with the active session ID."""
        with tracer.start_as_current_span("semantic_cache.set") as span:
            span.set_attribute("cache.query", query)
            if session_id:
                span.set_attribute("cache.session_id", session_id)

            query_vector = list(self.embedding_model.embed([query]))[0].tolist()
            seed_string = f"{session_id}_{query}" if session_id else query
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed_string))

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload={
                            "query": query,
                            "response": response,
                            "session_id": session_id or ""
                        }
                    )
                ]
            )
            span.set_attribute("cache.status", "STORED")

    def clear_session(self, session_id: str) -> None:
        """Deletes cache entries associated with a specific session ID."""
        if not session_id:
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            )
        except Exception:
            pass

    def clear_all(self) -> None:
        """Wipes the entire semantic cache collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
        except Exception:
            pass