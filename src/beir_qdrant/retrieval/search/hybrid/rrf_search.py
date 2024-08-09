import logging
import time
from collections import defaultdict
from typing import Dict

from qdrant_client import models
from tqdm import tqdm

from beir_qdrant.retrieval.search.hybrid import HybridQdrantSearch

logger = logging.getLogger(__name__)


class RRFHybridQdrantSearch(HybridQdrantSearch):
    """
    Hybrid search using Qdrant and FastEmbed searches and Rank Reciprocal Fusion (RRF) strategy.
    """

    def handle_queries(
        self, queries: Dict[str, str], limit: int
    ) -> Dict[str, Dict[str, float]]:
        query_ids, query_texts = zip(*queries.items())

        init_time = time.perf_counter()
        query_embeddings = [
            search.model.encode_queries(query_texts) for search in self.searches
        ]
        end_time = time.perf_counter()
        logger.info(
            f"Encoded {len(query_embeddings)} queries in {end_time - init_time:.8f} seconds"
        )

        # Convert the embeddings to the Qdrant format
        query_embeddings = [
            search.convert_embeddings_to_qdrant_format(embeddings)
            for search, embeddings in zip(self.searches, query_embeddings)
        ]

        results = defaultdict(dict)
        init_time = time.perf_counter()
        for query_idx, (query_id, query_embedding) in tqdm(
            enumerate(zip(query_ids, query_embeddings)),
            total=len(query_ids),
            desc="Queries",
        ):
            query_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(
                        query=query_embeddings[search_idx][query_idx],
                        using=search.vector_name,
                        limit=limit,
                        params=self.search_params,
                    )
                    for search_idx, search in enumerate(self.searches)
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
                search_params=self.search_params,
            )
            results[query_id] = {
                point.payload["doc_id"]: point.score for point in query_results.points
            }
        end_time = time.perf_counter()
        logger.info(
            f"Queried {self.collection_name} in {end_time - init_time:.8f} seconds"
        )

        return results
