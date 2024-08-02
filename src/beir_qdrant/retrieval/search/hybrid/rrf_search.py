from typing import List

from qdrant_client import models

from beir_qdrant.retrieval.search.hybrid import HybridQdrantSearch


class RRFHybridQdrantSearch(HybridQdrantSearch):
    """
    Hybrid search using Qdrant and FastEmbed searches and Rank Reciprocal Fusion (RRF) strategy.
    """

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        prefetch = [
            models.Prefetch(
                query=search.model.embed_query(query),
                using=search.vector_name,
                limit=limit,
                params=self.search_params,
            )
            for search in self.searches
        ]
        results = self.qdrant_client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=self.search_params,
        )
        return results.points
