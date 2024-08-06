from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.model_adapter.base import BaseDenseModelAdapter
from beir_qdrant.retrieval.search.qdrant import SingleNamedVectorQdrantBase


class DenseQdrantSearch(SingleNamedVectorQdrantBase, BaseSearch):
    """
    Dense search using Qdrant and FastEmbed model. By default, it uses all-miniLM-L6-v2 model for dense text embeddings.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseDenseModelAdapter,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 64,
        vector_name: str = "sparse",
        search_params: Optional[models.SearchParams] = None,
        distance: models.Distance = models.Distance.COSINE,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        on_disk: Optional[bool] = None,
        datatype: Optional[models.Datatype] = None,
    ):
        super().__init__(
            qdrant_client,
            model,  # noqa
            collection_name,
            initialize,
            clean_up,
            optimizers_config,
            batch_size,
            vector_name,
            search_params,
        )
        self.distance = distance
        self.hnsw_config = hnsw_config
        self.quantization_config = quantization_config
        self.on_disk = on_disk
        self.datatype = datatype

    def collection_config(self) -> Dict[str, Any]:
        test_embedding = self.model.embed_query("test")
        embedding_size = len(test_embedding)

        return dict(
            collection_name=self.collection_name,
            optimizers_config=self.optimizers_config,
            vectors_config={
                self.vector_name: models.VectorParams(
                    size=embedding_size,
                    distance=self.distance,
                    hnsw_config=self.hnsw_config,
                    quantization_config=self.quantization_config,
                    on_disk=self.on_disk,
                    datatype=self.datatype,
                )
            },
        )

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"distance={self.distance}",
            f"hnsw_config={self.hnsw_config}",
            f"quantization_config={self.quantization_config}",
            f"on_disk={self.on_disk}",
            f"datatype={self.datatype}",
        ]
