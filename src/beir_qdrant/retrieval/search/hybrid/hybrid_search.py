import abc
import uuid
from typing import Any, Dict, Iterable, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import QdrantBase, SingleVectorQdrantBase


class HybridQdrantSearch(QdrantBase, BaseSearch, abc.ABC):
    """
    Hybrid search using Qdrant and FastEmbed searches. By default, it uses all-miniLM-L6-v2 model for dense text
    embeddings and SPLADE model for sparse text embeddings. Both results are combined using a rank fusion algorithm.
    """

    BATCH_SIZE = 4

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        searches: Optional[Iterable[SingleVectorQdrantBase]] = None,
    ):
        super().__init__(qdrant_client, collection_name, initialize)
        self.inner_searches = searches or []

    def collection_config(self) -> Dict[str, Any]:
        # Get all the individual collection configs
        all_collection_configs = [
            search.collection_config() for search in self.inner_searches
        ]

        # Merge the configurations. Since we only use named vectors, we can safely merge them.
        merged_config = {
            "collection_name": self.collection_name,
            "vectors_config": {},
            "sparse_vectors_config": {},
        }
        for config in all_collection_configs:
            for vector_name, vector_params in config.get("vectors_config", {}).items():
                if vector_name in merged_config["vectors_config"]:
                    raise ValueError(f"Duplicate vector name: {vector_name}")
                merged_config["vectors_config"][vector_name] = vector_params
            for vector_name, vector_params in config.get(
                "sparse_vectors_config", {}
            ).items():
                if vector_name in merged_config["sparse_vectors_config"]:
                    raise ValueError(f"Duplicate sparse vector name: {vector_name}")
                merged_config["sparse_vectors_config"][vector_name] = vector_params

        return merged_config

    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        # Get all the individual points
        all_points = [
            search.doc_to_point(doc_id, doc) for search in self.inner_searches
        ]

        # Merge the points. Since we only use named vectors, we can safely merge them.
        merged_point = models.PointStruct(
            id=uuid.uuid4().hex,
            vector={},
            payload={"doc_id": doc_id, **doc},
        )
        for point in all_points:
            for vector_name, vector in point.vector.items():
                if vector_name in merged_point.vector:
                    raise ValueError(f"Duplicate vector name: {vector_name}")
                merged_point.vector[vector_name] = vector

        return merged_point

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"inner_searches=[{', '.join(map(str, self.inner_searches))}]",
        ]
