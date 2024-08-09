import logging
from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.models.base import BaseSparseModelAdapter
from beir_qdrant.retrieval.search.qdrant import SingleNamedVectorQdrantBase

logger = logging.getLogger(__name__)


class SparseQdrantSearch(SingleNamedVectorQdrantBase, BaseSearch):
    """
    Sparse search using Qdrant and FastEmbed model.
    """

    DEFAULT_BATCH_SIZE = 16

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseSparseModelAdapter,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 64,
        model_batch_size: Optional[int] = None,
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
            clean_up,
            optimizers_config,
            batch_size,
            model_batch_size,
            vector_name,
            search_params,
        )
        self.index = index
        self.modifier = modifier

    def collection_config(self) -> Dict[str, Any]:
        return dict(
            collection_name=self.collection_name,
            optimizers_config=self.optimizers_config,
            vectors_config={},
            sparse_vectors_config={
                self.vector_name: models.SparseVectorParams(
                    index=self.index,
                    modifier=self.modifier,
                )
            },
        )

    def convert_embeddings_to_qdrant_format(self, embeddings):
        """
        Convert sparse matrix rows to SparseVector instances.
        :param embeddings:
        :return:
        """
        return [
            models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.data.tolist(),
            )
            for embedding in embeddings
        ]

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"index={self.index}",
            f"modifier={self.modifier}",
        ]
