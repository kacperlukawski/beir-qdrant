import pytest

from beir_qdrant.retrieval.models.sentence_transformers import (
    SentenceTransformerModelAdapter,
    TokenEmbeddingsSentenceTransformerModelAdapter,
)


@pytest.fixture(scope="session")
def dense_model() -> SentenceTransformerModelAdapter:
    return SentenceTransformerModelAdapter(
        model_path="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture(scope="session")
def token_embeddings_model() -> TokenEmbeddingsSentenceTransformerModelAdapter:
    return TokenEmbeddingsSentenceTransformerModelAdapter("all-MiniLM-L6-v2")


def test_multi_vector_model_encode_queries_as_list_of_tensors(
    token_embeddings_model: TokenEmbeddingsSentenceTransformerModelAdapter,
):
    """
    Verify if the encode_queries method returns a numpy array with the queries of correct size.
    """
    embeddings = token_embeddings_model.encode_queries(
        [
            "Hello world, this is my first query!",
            "And here is another",
        ]
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert embeddings[0].shape[1] == 384
    assert embeddings[1].shape[1] == 384


def test_multi_vector_model_encode_corpus_as_list_of_tensors(
    token_embeddings_model: TokenEmbeddingsSentenceTransformerModelAdapter,
):
    """
    Verify if the encode_corpus method returns a numpy array with the queries of correct size.
    """
    embeddings = token_embeddings_model.encode_corpus(
        [
            {"text": "Hello world, this is my first query!"},
            {"text": "And here is another"},
            {"text": "The last sentence"},
        ]
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert embeddings[0].shape[1] == 384
    assert embeddings[1].shape[1] == 384
    assert embeddings[2].shape[1] == 384


def test_dense_model_casts_to_string_properly(
    dense_model: SentenceTransformerModelAdapter,
):
    model_str = str(dense_model)
    assert model_str is not None


def test_token_embeddings_model_casts_to_string_properly(
    token_embeddings_model: TokenEmbeddingsSentenceTransformerModelAdapter,
):
    model_str = str(token_embeddings_model)
    assert model_str is not None
