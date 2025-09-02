import uuid
from typing import Any, Dict, Iterable, Optional

from fastembed import LateInteractionTextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from beir_qdrant.retrieval.models.fastembed import (
    MultiVectorFastEmbedModelAdapter,
    MuveraPostprocessorAdapter,
)
from beir_qdrant.retrieval.search.qdrant import QdrantBase, batched


class MuveraReranking(QdrantBase):
    """
    Qdrant search with dual vectors: Muvera for initial search, multivector for reranking.
    Uses Qdrant's multi-stage search with prefetch.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        multivector_model_name: str,
        collection_name: str,
        oversample_factor: int = 5,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 64,
        model_batch_size: Optional[int] = None,
        k_sim: int = 4,
        dim_proj: int = 32,
        r_reps: int = 10,
        random_seed: int = 42,
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

        # Create the base multivector model
        try:
            import torch

            use_cuda = torch.cuda.is_available()
            providers = ["CUDAExecutionProvider"] if use_cuda else None
            base_model = LateInteractionTextEmbedding(
                model_name=multivector_model_name, providers=providers
            )
        except Exception:
            base_model = LateInteractionTextEmbedding(model_name=multivector_model_name)

        # Create Muvera adapter for initial retrieval
        self.muvera_model = MuveraPostprocessorAdapter(
            base_model,
            k_sim=k_sim,
            dim_proj=dim_proj,
            r_reps=r_reps,
            random_seed=random_seed,
        )

        # Create MultiVector adapter for reranking
        self.multivector_model = MultiVectorFastEmbedModelAdapter(
            multivector_model_name
        )

        # Store oversample factor
        self.oversample_factor = oversample_factor

    def collection_config(self) -> Dict[str, Any]:
        # Get test embeddings to determine sizes
        muvera_test = self.muvera_model.encode_queries(["test"])
        multivector_test = self.multivector_model.encode_queries(["test"])

        muvera_size = muvera_test.shape[1]
        multivector_size = multivector_test[0].shape[1]

        return dict(
            collection_name=self.collection_name,
            optimizers_config=self.optimizers_config,
            vectors_config={
                "muvera": models.VectorParams(
                    size=muvera_size,
                    distance=models.Distance.COSINE,
                ),
                "multivector": models.VectorParams(
                    size=multivector_size,
                    distance=models.Distance.COSINE,
                    # Disable HNSW for multivectors
                    hnsw_config=models.HnswConfigDiff(m=0),
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
        )

    def corpus_to_points(
        self, corpus: Dict[str, Dict[str, str]]
    ) -> Iterable[models.PointStruct]:
        corpus_items = list(corpus.items())

        num_total_batches = len(corpus_items) // self.model_batch_size + 1
        for corpus_batch in tqdm(
            batched(corpus_items, self.model_batch_size),
            total=num_total_batches,
            desc="Encoding corpus with dual vectors",
        ):
            document_ids, documents = zip(*corpus_batch)

            # Encode with both models
            muvera_embeddings = self.muvera_model.encode_corpus(
                documents, batch_size=self.model_batch_size
            )
            multivector_embeddings = self.multivector_model.encode_corpus(
                documents, batch_size=self.model_batch_size
            )

            # Convert multivector embeddings to Qdrant format
            multivector_embeddings = [
                [vector.tolist() for vector in embedding]
                for embedding in multivector_embeddings
            ]

            for doc_id, muvera_emb, multivector_emb in zip(
                document_ids, muvera_embeddings, multivector_embeddings
            ):
                yield models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={
                        "muvera": muvera_emb.tolist(),
                        "multivector": multivector_emb,
                    },
                    payload={"doc_id": doc_id, **corpus[doc_id]},
                )

    def handle_queries(
        self, queries: Dict[str, str], limit: int
    ) -> Dict[str, Dict[str, float]]:
        # Encode queries with both models
        muvera_query_embeddings = self.muvera_model.encode_queries(
            list(queries.values())
        )
        multivector_query_embeddings = self.multivector_model.encode_queries(
            list(queries.values())
        )

        # Use multi-stage search with prefetch
        results = {}
        for query_id, muvera_query_emb, multivector_query_emb in zip(
            queries.keys(), muvera_query_embeddings, multivector_query_embeddings
        ):
            # Multi-stage search: Muvera for initial retrieval, multivector for final ranking
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=models.Prefetch(
                    query=muvera_query_emb.tolist(),  # Muvera vector for initial retrieval
                    using="muvera",
                    limit=limit * self.oversample_factor,  # Oversample
                ),
                query=[
                    vector.tolist() for vector in multivector_query_emb
                ],  # Multivector for final ranking
                using="multivector",
                limit=limit,  # Final top_k results
                with_payload=True,
            )

            # Extract final results
            results[query_id] = {
                point.payload["doc_id"]: point.score for point in search_results.points
            }

        return results
