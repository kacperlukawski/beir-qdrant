import logging
import uuid
from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import SingleVectorQdrantBase

logger = logging.getLogger(__name__)


class SparseQdrantSearch(SingleVectorQdrantBase, BaseSearch):
    """
    Sparse search using Qdrant and FastEmbed model. By default, it uses SPLADE model for sparse text embeddings.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        vector_name: str = "sparse",
        sparse_model_name: str = "prithvida/Splade_PP_en_v1",
        index: Optional[models.SparseVectorParams] = None,
        modifier: Optional[models.Modifier] = None,
    ):
        super().__init__(qdrant_client, collection_name, initialize)
        self.vector_name = vector_name
        self.model = SparseTextEmbedding(model_name=sparse_model_name)
        self.index = index
        self.modifier = modifier

    def collection_config(self) -> Dict[str, Any]:
        return dict(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config={
                self.vector_name: models.SparseVectorParams(
                    index=self.index,
                    modifier=self.modifier,
                )
            },
        )

    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        doc_embedding = self.create_document_vector(doc["text"])
        return models.PointStruct(
            id=uuid.uuid4().hex,
            vector={self.vector_name: models.SparseVector(**doc_embedding)},
            payload={"doc_id": doc_id, **doc},
        )

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        query_embedding = self.create_query_vector(query)
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

    def create_document_vector(self, document: str) -> models.Vector:
        return next(self.model.passage_embed(document)).as_object()  # noqa

    def create_query_vector(self, query: str) -> models.Vector:
        return next(self.model.query_embed(query)).as_object()  # noqa

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"vector_name={self.vector_name}",
            f"sparse_model_name={self.model.model_name}",
            f"index={self.index}",
            f"modifier={self.modifier}",
        ]
