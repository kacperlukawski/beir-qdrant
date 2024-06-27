import uuid
from typing import List, Dict

from beir.retrieval.search import BaseSearch
from fastembed import TextEmbedding
from qdrant_client import models, QdrantClient

from beir_qdrant.retrieval.search.qdrant import QdrantBase


class DenseQdrantSearch(QdrantBase, BaseSearch):
    """
    Dense search using Qdrant and FastEmbed model. By default, it uses all-miniLM-L6-v2 model for dense text embeddings.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        vector_name: str = "sparse",
        initialize: bool = True,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(qdrant_client, collection_name, vector_name, initialize)
        self.model = TextEmbedding(model_name=dense_model_name)

    def recreate_collection(self):
        test_embedding = next(self.model.query_embed("test"))
        embedding_size = len(test_embedding)

        self.qdrant_client.delete_collection(self.collection_name)
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        doc_embedding = next(self.model.passage_embed([doc["text"]])).tolist()
        return models.PointStruct(
            id=uuid.uuid4().hex,
            vector={self.vector_name: doc_embedding},
            payload={"doc_id": doc_id, **doc},
        )

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        query_embedding = next(self.model.query_embed(query)).tolist()
        return self.qdrant_client.search(
            self.collection_name,
            query_vector=models.NamedVector(
                name=self.vector_name,
                vector=query_embedding,
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
