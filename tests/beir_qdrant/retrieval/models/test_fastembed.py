import numpy as np
import pytest
from fastembed import LateInteractionTextEmbedding

from beir_qdrant.retrieval.models.fastembed import (
    DenseFastEmbedModelAdapter,
    MultiVectorFastEmbedModelAdapter,
    MuveraPostprocessorAdapter,
    SparseFastEmbedModelAdapter,
)


class FakeLateInteractionTextEmbedding(LateInteractionTextEmbedding):
    """Fake LateInteractionTextEmbedding for testing that produces deterministic outputs."""

    def __init__(self):
        # Skip parent initialization to avoid loading actual models
        self.model_name = "fake-late-interaction-model"
        self._embedding_size = 128

    def passage_embed(self, texts, batch_size=8, **kwargs):
        """Generate fake embeddings for passages based on text content."""
        for text in texts:
            # Create deterministic embeddings based on text hash and length
            text_hash = hash(text) % 1000
            text_len = len(text)
            # Create a 2D array representing token-level embeddings
            num_tokens = min(max(text_len // 10, 3), 20)  # 3-20 tokens
            embedding_dim = 128

            # Generate deterministic values
            np.random.seed(text_hash)
            embedding = np.random.randn(num_tokens, embedding_dim).astype(np.float32)
            yield embedding

    def query_embed(self, queries, batch_size=16, **kwargs):
        """Generate fake embeddings for queries based on text content."""
        for query in queries:
            # Create deterministic embeddings based on text hash and length
            text_hash = hash(query) % 1000
            text_len = len(query)
            # Create a 2D array representing token-level embeddings
            num_tokens = min(max(text_len // 10, 2), 15)  # 2-15 tokens for queries
            embedding_dim = 128

            # Generate deterministic values
            np.random.seed(text_hash)
            embedding = np.random.randn(num_tokens, embedding_dim).astype(np.float32)
            yield embedding


@pytest.fixture(scope="session")
def dense_model() -> DenseFastEmbedModelAdapter:
    return DenseFastEmbedModelAdapter("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def sparse_model() -> SparseFastEmbedModelAdapter:
    return SparseFastEmbedModelAdapter("prithivida/Splade_PP_en_v1")


@pytest.fixture(scope="session")
def multi_vector_model() -> MultiVectorFastEmbedModelAdapter:
    return MultiVectorFastEmbedModelAdapter("colbert-ir/colbertv2.0")


@pytest.fixture(scope="session")
def muvera_adapter() -> MuveraPostprocessorAdapter:
    fake_model = FakeLateInteractionTextEmbedding()
    return MuveraPostprocessorAdapter(
        model=fake_model, k_sim=4, dim_proj=32, r_reps=10, random_seed=42
    )


def test_dense_model_encode_queries_as_numpy_array(
    dense_model: DenseFastEmbedModelAdapter,
):
    """
    Verify if the encode_queries method returns a numpy array with the queries of correct size.
    :param dense_model:
    :return:
    """
    embeddings = dense_model.encode_queries(
        [
            "Hello world, this is my first query!",
            "And here is another",
        ]
    )
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)


def test_dense_model_encode_corpus_as_numpy_array(
    dense_model: DenseFastEmbedModelAdapter,
):
    """
    Verify if the encode_corpus method returns a numpy array with the queries of correct size.
    :param dense_model:
    :return:
    """
    embeddings = dense_model.encode_corpus(
        [
            {"text": "Hello world, this is my first query!"},
            {"text": "And here is another"},
            {"text": "The last sentence"},
        ]
    )
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)


def test_sparse_model_encode_queries_as_sparse_array(
    sparse_model: SparseFastEmbedModelAdapter,
):
    """
    Verify if the encode_queries method returns a numpy array with the queries of correct size.
    :param sparse_model:
    :return:
    """
    embeddings = sparse_model.encode_queries(
        [
            "Hello world, this is my first query!",
            "And here is another",
        ]
    )
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 384


def test_sparse_model_encode_corpus_as_sparse_array(
    sparse_model: SparseFastEmbedModelAdapter,
):
    """
    Verify if the encode_corpus method returns a numpy array with the queries of correct size.
    :param sparse_model:
    :return:
    """
    embeddings = sparse_model.encode_corpus(
        [
            {"text": "Hello world, this is my first query!"},
            {"text": "And here is another"},
            {"text": "The last sentence"},
        ]
    )
    assert embeddings.shape[0] == 3
    assert embeddings.shape[1] > 384


def test_multi_vector_model_encode_queries_as_list_of_tensors(
    multi_vector_model: MultiVectorFastEmbedModelAdapter,
):
    """
    Verify if the encode_queries method returns a numpy array with the queries of correct size.
    :param multi_vector_model:
    :return:
    """
    embeddings = multi_vector_model.encode_queries(
        [
            "Hello world, this is my first query!",
            "And here is another",
        ]
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert embeddings[0].shape[1] == 128
    assert embeddings[1].shape[1] == 128


def test_multi_vector_model_encode_corpus_as_list_of_tensors(
    multi_vector_model: MultiVectorFastEmbedModelAdapter,
):
    """
    Verify if the encode_corpus method returns a numpy array with the queries of correct size.
    :param multi_vector_model:
    :return:
    """
    embeddings = multi_vector_model.encode_corpus(
        [
            {"text": "Hello world, this is my first query!"},
            {"text": "And here is another"},
            {"text": "The last sentence"},
        ]
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert embeddings[0].shape[1] == 128
    assert embeddings[1].shape[1] == 128
    assert embeddings[2].shape[1] == 128


def test_dense_model_casts_to_string_properly(
    dense_model: DenseFastEmbedModelAdapter,
):
    model_str = str(dense_model)
    assert model_str is not None


def test_sparse_model_casts_to_string_properly(
    sparse_model: SparseFastEmbedModelAdapter,
):
    model_str = str(sparse_model)
    assert model_str is not None


def test_multi_vector_model_casts_to_string_properly(
    multi_vector_model: MultiVectorFastEmbedModelAdapter,
):
    model_str = str(multi_vector_model)
    assert model_str is not None


def test_muvera_adapter_encode_corpus_consistency_and_differentiation(
    muvera_adapter: MuveraPostprocessorAdapter,
):
    """
    Test MuveraPostprocessorAdapter encode_corpus method for consistency and differentiation.
    Same inputs should produce same outputs, different inputs should produce different outputs.
    """
    corpus1 = [
        {"text": "Hello world, this is my first document!"},
        {"text": "And here is another document"},
        {"text": "The last document in the corpus"},
    ]

    corpus2 = [
        {"text": "Different content here"},
        {"text": "Another different document"},
        {"text": "Yet another unique text"},
    ]

    # Test consistency: same input should produce same output
    embeddings1_first = muvera_adapter.encode_corpus(corpus1)
    embeddings1_second = muvera_adapter.encode_corpus(corpus1)

    assert isinstance(embeddings1_first, np.ndarray)
    assert isinstance(embeddings1_second, np.ndarray)
    assert embeddings1_first.shape == embeddings1_second.shape
    assert np.allclose(
        embeddings1_first, embeddings1_second
    ), "Same corpus should produce identical embeddings"

    # Test differentiation: different inputs should produce different outputs
    embeddings2 = muvera_adapter.encode_corpus(corpus2)

    assert isinstance(embeddings2, np.ndarray)
    assert (
        embeddings1_first.shape[0] == embeddings2.shape[0]
    )  # Same number of documents
    assert not np.allclose(
        embeddings1_first, embeddings2
    ), "Different corpus should produce different embeddings"

    # Test output dimensions are reasonable (should produce dense vectors)
    assert (
        len(embeddings1_first.shape) == 2
    ), f"Expected 2D array, got shape {embeddings1_first.shape}"
    assert (
        embeddings1_first.shape[0] == 3
    ), f"Expected 3 documents, got {embeddings1_first.shape[0]}"
    assert (
        embeddings1_first.shape[1] > 0
    ), f"Expected positive embedding dimension, got {embeddings1_first.shape[1]}"


def test_muvera_adapter_encode_queries_consistency_and_differentiation(
    muvera_adapter: MuveraPostprocessorAdapter,
):
    """
    Test MuveraPostprocessorAdapter encode_queries method for consistency and differentiation.
    Same inputs should produce same outputs, different inputs should produce different outputs.
    """
    queries1 = [
        "Hello world, this is my first query!",
        "And here is another query",
    ]

    queries2 = [
        "Different query content here",
        "Another completely different query",
    ]

    # Test consistency: same input should produce same output
    embeddings1_first = muvera_adapter.encode_queries(queries1)
    embeddings1_second = muvera_adapter.encode_queries(queries1)

    assert isinstance(embeddings1_first, np.ndarray)
    assert isinstance(embeddings1_second, np.ndarray)
    assert embeddings1_first.shape == embeddings1_second.shape
    assert np.allclose(
        embeddings1_first, embeddings1_second
    ), "Same queries should produce identical embeddings"

    # Test differentiation: different inputs should produce different outputs
    embeddings2 = muvera_adapter.encode_queries(queries2)

    assert isinstance(embeddings2, np.ndarray)
    assert embeddings1_first.shape[0] == embeddings2.shape[0]  # Same number of queries
    assert not np.allclose(
        embeddings1_first, embeddings2
    ), "Different queries should produce different embeddings"

    # Test output dimensions are reasonable (should produce dense vectors)
    assert (
        len(embeddings1_first.shape) == 2
    ), f"Expected 2D array, got shape {embeddings1_first.shape}"
    assert (
        embeddings1_first.shape[0] == 2
    ), f"Expected 2 queries, got {embeddings1_first.shape[0]}"
    assert (
        embeddings1_first.shape[1] > 0
    ), f"Expected positive embedding dimension, got {embeddings1_first.shape[1]}"
