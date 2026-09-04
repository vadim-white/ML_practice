"""Обёртка над qdrant-client: хранение item-эмбеддингов и ANN-поиск.

Отдельная коллекция на каждый метод ("als_items", "two_tower_items"), чтобы
ALS и Two-Tower можно было сравнивать независимо на этапе 4.
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, name: str, dim: int) -> None:
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    def upsert_items(self, name: str, item_ids: list, embeddings) -> None:
        points = [
            qmodels.PointStruct(id=int(item_id), vector=vector.tolist())
            for item_id, vector in zip(item_ids, embeddings)
        ]
        self.client.upsert(collection_name=name, points=points)

    def search(self, name: str, query_vector, top_k: int = 100):
        hits = self.client.search(
            collection_name=name,
            query_vector=query_vector.tolist(),
            limit=top_k,
        )
        return [(hit.id, hit.score) for hit in hits]
