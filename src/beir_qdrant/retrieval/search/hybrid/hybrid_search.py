import abc
import uuid
from typing import Any, Dict, Iterable, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import QdrantBase, SingleNamedVectorQdrantBase


class HybridQdrantSearch(QdrantBase, BaseSearch, abc.ABC):
    """
    Hybrid search using Qdrant and FastEmbed searches. By default, it uses all-miniLM-L6-v2 model for dense text
    embeddings and SPLADE model for sparse text embeddings. Both results are combined using a rank fusion algorithm.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 4,
        model_batch_size: Optional[int] = None,
        searches: Optional[Iterable[SingleNamedVectorQdrantBase]] = None,
        search_params: Optional[models.SearchParams] = None,
    ):
        super().__init__(
            qdrant_client,
            collection_name,
            initialize,
            clean_up,
            optimizers_config,
            batch_size,
            model_batch_size,
        )
        self.searches = searches or []
        self.search_params = search_params

    def collection_config(self) -> Dict[str, Any]:
        # Get all the individual collection configs
        all_collection_configs = [
            search.collection_config() for search in self.searches
        ]

        # Merge the configurations. Since we only use named vectors, we can safely merge them.
        merged_config = {
            "collection_name": self.collection_name,
            "optimizers_config": self.optimizers_config,
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

    def corpus_to_points(
        self, corpus: Dict[str, Dict[str, str]]
    ) -> Iterable[models.PointStruct]:
        for subpoints in zip(
            *[search.corpus_to_points(corpus) for search in self.searches]
        ):
            merged_vectors = {}
            for point in subpoints:
                for vector_name, vector in point.vector.items():
                    merged_vectors[vector_name] = vector

            merged_point = models.PointStruct(
                id=uuid.uuid4().hex,
                vector=merged_vectors,
                payload=subpoints[0].payload,
            )
            yield merged_point

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"searches=[{', '.join(map(str, self.searches))}]",
        ]
