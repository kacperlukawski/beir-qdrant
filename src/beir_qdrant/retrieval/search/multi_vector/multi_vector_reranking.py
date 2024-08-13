from collections import defaultdict
from typing import Dict

import numpy as np
from beir.retrieval.search import BaseSearch
from beir.util import cos_sim
from torch import Tensor
from tqdm import tqdm

from beir_qdrant.retrieval.models.base import BaseMultiVectorModelAdapter
from beir_qdrant.retrieval.search.qdrant import QdrantBase


class MultiVectorReranking(BaseSearch):
    """
    This is a multi-vector reranking pipeline that oversample the search results and then rerank them using an
    additional oversample_model. Multi-vectors are not stored in Qdrant, but are generated on-the-fly using the oversample_model. That saves
    storage space and allows for more flexible reranking strategies.

    The primary search is done using a Qdrant search, and the reranking is done using a multi-vector oversample_model with MaxSim
    reranking strategy, similar to ColBERT.
    """

    def __init__(
        self,
        oversample_search: QdrantBase,
        rerank_model: BaseMultiVectorModelAdapter,
        oversample_factor: int = 5,
    ):
        self.oversample_search = oversample_search
        self.rerank_model = rerank_model
        self.oversample_factor = oversample_factor
        self.clean_up = oversample_search.clean_up

        # We overwrite the value of clean up here, to avoid the collection being removed from Qdrant. We still need to
        # access the points and do the reranking. If the cleanup was requested, this class will handle this on its own.
        oversample_search.clean_up = False

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        # Use the underlying Qdrant search to get the initial results (oversampled)
        results = self.oversample_search.search(
            corpus, queries, top_k * self.oversample_factor, **kwargs
        )

        # Vectorize the queries using the multi-vector oversample_model
        query_embeddings = self.rerank_model.encode_queries(list(queries.values()))

        # Rerank the results using the multi-vector oversample_model
        reranked_results = defaultdict(dict)
        for query_id, query_embedding in zip(
            tqdm(queries.keys(), desc="Reranking queries"), query_embeddings
        ):
            reranked_results[query_id] = self._rerank_query_results(
                query_embedding, results[query_id], top_k
            )

        # Clean up, if that was requested in the oversample search
        if self.clean_up:
            self.oversample_search.qdrant_client.delete_collection(
                self.oversample_search.collection_name
            )

        return reranked_results

    def _rerank_query_results(
        self, query_embedding: Tensor, query_results: Dict[str, float], top_k: int
    ) -> Dict[str, float]:
        document_ids, _ = zip(*query_results.items())

        # Load full documents from the Qdrant search, to be able to encode them with the multi-vector model
        documents = [
            self.oversample_search.get_by_document_id(doc_id) for doc_id in document_ids
        ]

        # Calculate the document embeddings using the multi-vector model
        document_embeddings = self.rerank_model.encode_corpus(documents)

        # Calculate the MaxSim score for each document
        document_scores = {}
        for doc_id, doc_embedding in zip(document_ids, document_embeddings):
            document_scores[doc_id] = self._max_sim(query_embedding, doc_embedding)

        return dict(
            sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

    def _max_sim(self, query_embeddings: Tensor, document_embeddings: Tensor) -> float:
        """
        Calculate MaxSim score, similarly to what ColBERT does.
        """
        similarities = cos_sim(query_embeddings, document_embeddings).cpu().numpy()
        max_similarities = np.max(similarities, axis=1)
        return np.sum(max_similarities).tolist()

    def __str__(self) -> str:
        return (
            f"MultiVectorReranking("
            f"oversample_search={self.oversample_search}, "
            f"rerank_model={self.rerank_model}, "
            f"oversample_factor={self.oversample_factor}, "
            f"clean_up={self.clean_up}"
            f")"
        )
