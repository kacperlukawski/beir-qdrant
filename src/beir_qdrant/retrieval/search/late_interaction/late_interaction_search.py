import uuid
from typing import Any, Dict, List, Optional

from beir.retrieval.search import BaseSearch
from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import SingleVectorQdrantBase


class LateInteractionQdrantSearch(SingleVectorQdrantBase, BaseSearch):
    """
    Late interaction search using Qdrant, for searches like COLBERT.
    """

    BATCH_SIZE = 4

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        vector_name: str = "late_interaction",
        late_interaction_model_name: str = "colbert-ir/colbertv2.0",
        distance: models.Distance = models.Distance.COSINE,
        hnsw_config: Optional[models.HnswConfigDiff] = None,
        quantization_config: Optional[models.QuantizationConfig] = None,
        on_disk: Optional[bool] = None,
        datatype: Optional[models.Datatype] = None,
    ):
        super().__init__(qdrant_client, collection_name, initialize)
        self.vector_name = vector_name
        self.model = LateInteractionTextEmbedding(
            model_name=late_interaction_model_name
        )
        self.distance = distance
        self.hnsw_config = hnsw_config
        self.quantization_config = quantization_config
        self.on_disk = on_disk
        self.datatype = datatype

    def collection_config(self) -> Dict[str, Any]:
        test_embedding = next(self.model.query_embed("test"))
        embedding_size = test_embedding.shape[1]

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

    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        doc_embedding = self.create_document_vector(doc["text"])
        return models.PointStruct(
            id=uuid.uuid4().hex,
            vector={self.vector_name: doc_embedding},
            payload={"doc_id": doc_id, **doc},
        )

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        query_embedding = self.create_query_vector(query)
        query_response = self.qdrant_client.query_points(
            self.collection_name,
            query=query_embedding,
            using=self.vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return query_response.points

    def create_document_vector(self, document: str) -> models.Vector:
        return next(self.model.passage_embed([document])).tolist()  # noqa

    def create_query_vector(self, query: str) -> models.Vector:
        return next(self.model.query_embed(query)).tolist()  # noqa

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"vector_name={self.vector_name}",
            f"late_interaction_model_name={self.model.model_name}",
            f"distance={self.distance}",
            f"hnsw_config={self.hnsw_config}",
            f"quantization_config={self.quantization_config}",
            f"on_disk={self.on_disk}",
            f"datatype={self.datatype}",
        ]
