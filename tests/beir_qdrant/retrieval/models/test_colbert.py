import pytest

from beir_qdrant.retrieval.models.colbert import ColbertModelAdapter

QUERY_MAXLEN = 32
DOC_MAXLEN = 512


@pytest.fixture(scope="session")
def colbert_model() -> ColbertModelAdapter:
    return ColbertModelAdapter(
        "colbert-ir/colbertv2.0", query_maxlen=QUERY_MAXLEN, doc_maxlen=DOC_MAXLEN
    )


@pytest.mark.parametrize("batch_size", (1, 2))
def test_encode_queries(colbert_model: ColbertModelAdapter, batch_size: int):
    queries = ["What is the capital of France?", "Who is the president of the USA?"]
    embeddings = colbert_model.encode_queries(queries, batch_size=batch_size)
    assert isinstance(embeddings, list)
    assert embeddings[0].shape == (QUERY_MAXLEN, 128)
    assert embeddings[1].shape == (QUERY_MAXLEN, 128)


@pytest.mark.parametrize("batch_size", (1, 2))
def test_encode_corpus(colbert_model: ColbertModelAdapter, batch_size: int):
    corpus = [
        {"id": "1", "text": "Paris is the capital of France."},
        {"id": "2", "text": "Washington is the capital of the USA."},
    ]
    embeddings = colbert_model.encode_corpus(corpus, batch_size=batch_size)
    assert isinstance(embeddings, list)
    assert embeddings[0].shape == (11, 128)
    assert embeddings[1].shape == (11, 128)


def test_colbert_model_casts_to_string_properly(colbert_model: ColbertModelAdapter):
    model_str = str(colbert_model)
    assert model_str is not None
