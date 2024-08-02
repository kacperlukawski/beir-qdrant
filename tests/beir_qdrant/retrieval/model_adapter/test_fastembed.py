import pytest

from beir_qdrant.retrieval.model_adapter.fastembed import DenseFastEmbedModelAdapter


@pytest.fixture(scope="session")
def dense_model() -> DenseFastEmbedModelAdapter:
    return DenseFastEmbedModelAdapter("sentence-transformers/all-MiniLM-L6-v2")


def test_dense_model_returns_python_list(dense_model: DenseFastEmbedModelAdapter):
    """
    Verify if the embed_documents method returns a Python list, not a numpy array or tensor.
    :param dense_model:
    :return:
    """
    embeddings = dense_model.embed_documents(["Hello, world!"])[0]
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, float) for embedding in embeddings)
