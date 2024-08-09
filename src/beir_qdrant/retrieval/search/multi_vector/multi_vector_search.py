from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.models.base import BaseMultiVectorModelAdapter
from beir_qdrant.retrieval.search.qdrant import SingleNamedVectorQdrantBase


class MultiVectorQdrantSearch(SingleNamedVectorQdrantBase, BaseSearch):
    """
    Multi-vector search, for example for late interaction models like ColBERT.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseMultiVectorModelAdapter,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 4,
        model_batch_size: Optional[int] = None,
        vector_name: str = "multi_vector",
        search_params: Optional[models.SearchParams] = None,
        distance: models.Distance = models.Distance.COSINE,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        on_disk: Optional[bool] = None,
        datatype: Optional[models.Datatype] = None,
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
        self.distance = distance
        self.hnsw_config = hnsw_config
        self.quantization_config = quantization_config
        self.on_disk = on_disk
        self.datatype = datatype

    def collection_config(self) -> Dict[str, Any]:
        assert isinstance(self.model, BaseMultiVectorModelAdapter)
        test_embedding = self.model.encode_queries(["test"])
        embedding_size = test_embedding[0].shape[1]

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
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )

    def convert_embeddings_to_qdrant_format(self, embeddings):
        return [[vector.tolist() for vector in embedding] for embedding in embeddings]

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"distance={self.distance}",
            f"hnsw_config={self.hnsw_config}",
            f"quantization_config={self.quantization_config}",
            f"on_disk={self.on_disk}",
            f"datatype={self.datatype}",
        ]
