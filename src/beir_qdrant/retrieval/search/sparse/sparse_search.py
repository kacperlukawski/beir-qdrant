import logging
import uuid
from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.model_adapter.base import BaseSparseModelAdapter
from beir_qdrant.retrieval.search.qdrant import SingleNamedVectorQdrantBase

logger = logging.getLogger(__name__)


class SparseQdrantSearch(SingleNamedVectorQdrantBase, BaseSearch):
    """
    Sparse search using Qdrant and FastEmbed model. By default, it uses SPLADE model for sparse text embeddings.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseSparseModelAdapter,
        collection_name: str,
        initialize: bool = True,
        vector_name: str = "sparse",
        index: Optional[models.SparseVectorParams] = None,
        modifier: Optional[models.Modifier] = None,
    ):
        super().__init__(qdrant_client, model, collection_name, initialize)  # noqa
        self.model = model
        self.vector_name = vector_name
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
        doc_embedding = self.model.embed_document(doc["text"])
        return models.PointStruct(
            id=uuid.uuid4().hex,
            vector={self.vector_name: doc_embedding},
            payload={"doc_id": doc_id, **doc},
        )

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"model={self.model}",
            f"vector_name={self.vector_name}",
            f"index={self.index}",
            f"modifier={self.modifier}",
        ]
