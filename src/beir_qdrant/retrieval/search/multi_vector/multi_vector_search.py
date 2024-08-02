from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.model_adapter.base import BaseMultiVectorModelAdapter
from beir_qdrant.retrieval.search.qdrant import SingleNamedVectorQdrantBase


class MultiVectorQdrantSearch(SingleNamedVectorQdrantBase, BaseSearch):
    """
    Multi-vector search, for example for late interaction models like ColBERT.
    """

    BATCH_SIZE = 4

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseMultiVectorModelAdapter,
        collection_name: str,
        initialize: bool = True,
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
            model,  # noqa
            collection_name,
            initialize,
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
        test_embedding = self.model.embed_query("test")
        embedding_size = len(test_embedding[0])

        return dict(
            collection_name=self.collection_name,
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

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"model={self.model}",
            f"vector_name={self.vector_name}",
            f"distance={self.distance}",
            f"hnsw_config={self.hnsw_config}",
            f"quantization_config={self.quantization_config}",
            f"on_disk={self.on_disk}",
            f"datatype={self.datatype}",
        ]
