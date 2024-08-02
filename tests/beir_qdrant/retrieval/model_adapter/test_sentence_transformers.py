import pytest

from beir_qdrant.retrieval.model_adapter.sentence_transformers import (
    TokenEmbeddingsSentenceTransformerModelAdapter,
)


@pytest.fixture(scope="session")
def token_embeddings_model() -> TokenEmbeddingsSentenceTransformerModelAdapter:
    return TokenEmbeddingsSentenceTransformerModelAdapter("all-MiniLM-L6-v2")


def test_embed_documents_returns_python_list(
    token_embeddings_model: TokenEmbeddingsSentenceTransformerModelAdapter,
):
    """
    Verify if the embed_documents method returns a Python list, not a numpy array or tensor.
    :param token_embeddings_model:
    :return:
    """
    embeddings = token_embeddings_model.embed_documents(["Hello, world!"])[0]
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    for embedding in embeddings:
        assert all(isinstance(value, float) for value in embedding)
