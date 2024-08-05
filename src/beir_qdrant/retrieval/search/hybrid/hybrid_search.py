import abc
import uuid
from typing import Any, Dict, Iterable, List, Optional

from beir.retrieval.search import BaseSearch
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.search.qdrant import (
    Document,
    QdrantBase,
    SingleNamedVectorQdrantBase,
)


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
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        searches: Optional[Iterable[SingleNamedVectorQdrantBase]] = None,
        search_params: Optional[models.SearchParams] = None,
    ):
        super().__init__(
            qdrant_client, collection_name, initialize, clean_up, optimizers_config
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

    def docs_to_points(self, documents: List[Document]) -> List[models.PointStruct]:
        if len(self.searches) == 0:
            raise ValueError("No search models are provided")

        # Create embeddings for all the documents in a single model call
        texts = [doc.payload["text"] for doc in documents]
        embeddings = {
            search.vector_name: search.model.embed_documents(texts)
            for search in self.searches
        }

        points = []
        for doc_idx, doc in enumerate(documents):
            merged_point = models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    vector_name: embedding[doc_idx]
                    for vector_name, embedding in embeddings.items()
                },
                payload={"doc_id": doc.id, **doc.payload},
            )
            points.append(merged_point)
        return points

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"searches=[{', '.join(map(str, self.searches))}]",
        ]
