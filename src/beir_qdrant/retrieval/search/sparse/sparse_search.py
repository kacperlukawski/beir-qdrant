import logging
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
        search_params: Optional[models.SearchParams] = None,
        index: Optional[models.SparseVectorParams] = None,
        modifier: Optional[models.Modifier] = None,
    ):
        super().__init__(
            qdrant_client,
            model,
            collection_name,
            initialize,
            vector_name,
            search_params,
        )  # noqa
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

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"model={self.model}",
            f"vector_name={self.vector_name}",
            f"index={self.index}",
            f"modifier={self.modifier}",
        ]
