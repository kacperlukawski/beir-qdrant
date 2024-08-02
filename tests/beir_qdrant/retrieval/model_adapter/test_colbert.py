import pytest

from beir_qdrant.retrieval.model_adapter.colbert import ColbertModelAdapter


@pytest.fixture(scope="session")
def colbert_model() -> ColbertModelAdapter:
    return ColbertModelAdapter("colbert-ir/colbertv2.0")


def test_colbert_embed_documents_returns_python_list(
    colbert_model: ColbertModelAdapter,
):
    """
    Verify if the embed_documents method returns a Python list, not a numpy array or tensor.
    :param colbert_model:
    :return:
    """
    embeddings = colbert_model.embed_documents(
        ["Hello, world of late interaction models!"]
    )[0]
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    for embedding in embeddings:
        assert all(isinstance(value, float) for value in embedding)


def test_colbert_embed_query_returns_python_list(
    colbert_model: ColbertModelAdapter,
):
    """
    Verify if the embed_query method returns a list of lists.
    :param colbert_model:
    :return:
    """
    embeddings = colbert_model.embed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert all(isinstance(embedding, list) for embedding in embeddings)
    for embedding in embeddings:
        assert all(isinstance(value, float) for value in embedding)
