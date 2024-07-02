import logging
import uuid
from typing import Any, Dict, List

from beir.retrieval.search import BaseSearch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import QdrantBase

logger = logging.getLogger(__name__)


class SparseQdrantSearch(QdrantBase, BaseSearch):
    """
    Sparse search using Qdrant and FastEmbed model. By default, it uses SPLADE model for sparse text embeddings.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        vector_name: str = "sparse",
        initialize: bool = True,
        sparse_model_name: str = "prithvida/Splade_PP_en_v1",
    ):
        super().__init__(qdrant_client, collection_name, vector_name, initialize)
        self.model = SparseTextEmbedding(model_name=sparse_model_name)

    def collection_config(self) -> Dict[str, Any]:
        return dict(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config={
                self.vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        doc_embedding = next(self.model.embed(doc["text"])).as_object()
        return models.PointStruct(
            id=uuid.uuid4().hex,
            vector={self.vector_name: models.SparseVector(**doc_embedding)},
            payload={"doc_id": doc_id, **doc},
        )

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        query_embedding = next(self.model.embed(query)).as_object()
        return self.qdrant_client.search(
            self.collection_name,
            query_vector=models.NamedSparseVector(
                name=self.vector_name,
                vector=models.SparseVector(**query_embedding),
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

    def _str_params(self) -> List[str]:
        return super()._str_params() + [f"sparse_model_name={self.model.model_name}"]
