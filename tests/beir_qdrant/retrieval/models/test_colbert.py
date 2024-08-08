import pytest

from beir_qdrant.retrieval.models.colbert import ColbertModelAdapter


@pytest.fixture(scope="session")
def colbert_model() -> ColbertModelAdapter:
    return ColbertModelAdapter("colbert-ir/colbertv2.0")


# TODO: write tests for ColBERT model
