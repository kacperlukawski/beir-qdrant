import uuid
from typing import Any, Dict, List

from beir.retrieval.search import BaseSearch
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import QdrantBase


class LateInteractionQdrantSearch(QdrantBase, BaseSearch):
    """
    Late interaction search using Qdrant, for models like COLBERT.
    """

    BATCH_SIZE = 4

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        vector_name: str = "late_interaction",
        initialize: bool = True,
        late_interaction_model_name: str = "colbert-ir/colbertv2.0",
    ):
        super().__init__(qdrant_client, collection_name, vector_name, initialize)
        self.model = LateInteractionTextEmbedding(
            model_name=late_interaction_model_name
        )

    def collection_config(self) -> Dict[str, Any]:
        test_embedding = next(self.model.query_embed("test"))
        embedding_size = test_embedding.shape[1]

        return dict(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
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
        query_response = self.qdrant_client.query_points(
            self.collection_name,
            query=query_embedding,
            using=self.vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return query_response.points
