from qdrant_client import QdrantClient

# Single client instance shared across all modules
qdrant_client = QdrantClient(path="./qdrant_db")