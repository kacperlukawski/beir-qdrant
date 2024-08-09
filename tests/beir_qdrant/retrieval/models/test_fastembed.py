import numpy as np
import pytest

from beir_qdrant.retrieval.models.fastembed import (
    DenseFastEmbedModelAdapter,
    MultiVectorFastEmbedModelAdapter,
    SparseFastEmbedModelAdapter,
)


@pytest.fixture(scope="session")
def dense_model() -> DenseFastEmbedModelAdapter:
    return DenseFastEmbedModelAdapter("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def sparse_model() -> SparseFastEmbedModelAdapter:
    return SparseFastEmbedModelAdapter("prithvida/SPLADE_PP_en_v1")


@pytest.fixture(scope="session")
def multi_vector_model() -> MultiVectorFastEmbedModelAdapter:
    return MultiVectorFastEmbedModelAdapter("colbert-ir/colbertv2.0")


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
