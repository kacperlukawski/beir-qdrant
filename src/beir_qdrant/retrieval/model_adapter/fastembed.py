from typing import List

from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from beir_qdrant.retrieval.model_adapter.base import (
    BaseDenseModelAdapter,
    BaseMultiVectorModelAdapter,
    BaseSparseModelAdapter,
    DenseVector,
)


class DenseFastEmbedModelAdapter(BaseDenseModelAdapter):
    """
    Adapter for the FastEmbed text embedding models.
    """

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_document(self, document: str) -> DenseVector:
        return next(self._model.passage_embed([document])).tolist()  # noqa

    def embed_query(self, query: str) -> DenseVector:
        return next(self._model.query_embed(query)).tolist()  # noqa


class SparseFastEmbedModelAdapter(BaseSparseModelAdapter):
    """
    Adapter for the FastEmbed sparse text embedding models.
    """

    def __init__(self, model_name: str):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_document(self, document: str) -> models.SparseVector:
        embedding = next(self._model.passage_embed([document])).as_object()  # noqa
        return models.SparseVector(**embedding)

    def embed_query(self, query: str) -> models.SparseVector:
        embedding = next(self._model.query_embed(query)).as_object()  # noqa
        return models.SparseVector(**embedding)


class MultiVectorFastEmbedModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for the FastEmbed late interaction models.
    """

    def __init__(self, model_name: str):
        self._model = LateInteractionTextEmbedding(model_name=model_name)

    def embed_document(self, document: str) -> List[DenseVector]:
        return next(self._model.passage_embed([document])).tolist()  # noqa

    def embed_query(self, query: str) -> List[DenseVector]:
        return next(self._model.query_embed(query)).tolist()  # noqa
