from pathlib import Path

import pytest
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient

from beir_qdrant.retrieval.models.fastembed import (
    DenseFastEmbedModelAdapter,
    MultiVectorFastEmbedModelAdapter,
    SparseFastEmbedModelAdapter,
)
from beir_qdrant.retrieval.search.dense import DenseQdrantSearch
from beir_qdrant.retrieval.search.hybrid import RRFHybridQdrantSearch
from beir_qdrant.retrieval.search.multi_vector import MultiVectorQdrantSearch
from beir_qdrant.retrieval.search.sparse import SparseQdrantSearch

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# TODO: consider adding some test implementations of the models to lower the time of the tests


@pytest.fixture(scope="session")
def data_loader() -> GenericDataLoader:
    return GenericDataLoader(data_folder=str(FIXTURES_DIR / "datasets" / "test"))


@pytest.fixture(scope="session")
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture(scope="session")
def dense_model() -> DenseFastEmbedModelAdapter:
    return DenseFastEmbedModelAdapter(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture(scope="session")
def sparse_model() -> SparseFastEmbedModelAdapter:
    return SparseFastEmbedModelAdapter(model_name="prithvida/Splade_PP_en_v1")


@pytest.fixture(scope="session")
def multi_vector_model() -> MultiVectorFastEmbedModelAdapter:
    return MultiVectorFastEmbedModelAdapter(model_name="colbert-ir/colbertv2.0")


def test_search_indexes_all_documents(
    qdrant_client: QdrantClient, dense_model: DenseFastEmbedModelAdapter
):
    model = DenseQdrantSearch(
        qdrant_client,
        model=dense_model,
        collection_name="my-collection",
        vector_name="vector",
        initialize=True,
        clean_up=False,
    )

    corpus = {
        "1": {"id": "1", "text": "Paris is the capital of France."},
        "2": {"id": "2", "text": "Washington is the capital of the USA."},
    }
    queries = {
        "1": "What is the capital of France?",
    }
    model.search(corpus, queries, top_k=1)

    assert qdrant_client.count("my-collection").count == 2


@pytest.mark.integration
def test_dense_model_search_integration(
    data_loader: GenericDataLoader,
    qdrant_client: QdrantClient,
    dense_model: DenseFastEmbedModelAdapter,
):
    from beir_qdrant.retrieval.search.dense import DenseQdrantSearch

    corpus, queries, qrels = data_loader.load(split="test")

    model = DenseQdrantSearch(
        qdrant_client,
        model=dense_model,
        collection_name="my-collection",
        vector_name="vector",
        initialize=True,
        clean_up=True,
    )

    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[2])

    assert "NDCG@2" in ndcg
    assert "MAP@2" in _map
    assert "Recall@2" in recall
    assert "P@2" in precision


@pytest.mark.integration
def test_sparse_model_search_integration(
    data_loader: GenericDataLoader,
    qdrant_client: QdrantClient,
    sparse_model: SparseFastEmbedModelAdapter,
):
    corpus, queries, qrels = data_loader.load(split="test")

    model = SparseQdrantSearch(
        qdrant_client,
        model=sparse_model,
        collection_name="my-collection",
        vector_name="vector",
        initialize=True,
        clean_up=True,
    )

    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[2])

    assert "NDCG@2" in ndcg
    assert "MAP@2" in _map
    assert "Recall@2" in recall
    assert "P@2" in precision


@pytest.mark.integration
def test_rrf_search_integration(
    data_loader: GenericDataLoader,
    qdrant_client: QdrantClient,
    dense_model: DenseFastEmbedModelAdapter,
    sparse_model: SparseFastEmbedModelAdapter,
):
    corpus, queries, qrels = data_loader.load(split="test")

    dense_search = RRFHybridQdrantSearch(
        qdrant_client,
        searches=[
            DenseQdrantSearch(
                qdrant_client,
                model=dense_model,
                collection_name="my-collection",
                vector_name="dense-vector",
            ),
            SparseQdrantSearch(
                qdrant_client,
                model=sparse_model,
                collection_name="my-collection",
                vector_name="sparse-vector",
            ),
        ],
        collection_name="my-collection",
        initialize=True,
        clean_up=True,
    )

    retriever = EvaluateRetrieval(dense_search)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[2])

    assert "NDCG@2" in ndcg
    assert "MAP@2" in _map
    assert "Recall@2" in recall
    assert "P@2" in precision


@pytest.mark.integration
def test_multi_vector_search_integration(
    data_loader: GenericDataLoader,
    qdrant_client: QdrantClient,
    multi_vector_model: MultiVectorFastEmbedModelAdapter,
):
    corpus, queries, qrels = data_loader.load(split="test")

    model = MultiVectorQdrantSearch(
        qdrant_client,
        model=multi_vector_model,
        collection_name="my-collection",
        vector_name="vector",
        initialize=True,
        clean_up=True,
    )

    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[2])

    assert "NDCG@2" in ndcg
    assert "MAP@2" in _map
    assert "Recall@2" in recall
    assert "P@2" in precision
