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
                query=search.create_query_vector(query),
                using=search.vector_name,
                limit=limit,
            )
            for search in self.inner_searches
        ]
        results = self.qdrant_client.query_points(
            self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=limit,
        )
        return results.points
