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

    def embed_documents(self, documents: List[str]) -> List[DenseVector]:
        embeddings = self._model.passage_embed(documents)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> DenseVector:
        return next(self._model.query_embed(query)).tolist()  # noqa

    def __str__(self):
        return f"DenseFastEmbedModelAdapter(model_name={self._model.model_name})"


class SparseFastEmbedModelAdapter(BaseSparseModelAdapter):
    """
    Adapter for the FastEmbed sparse text embedding models.
    """

    def __init__(self, model_name: str):
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_document(self, document: str) -> models.SparseVector:
        embedding = next(self._model.passage_embed([document])).as_object()  # noqa
        return models.SparseVector(**embedding)

    def embed_documents(self, documents: List[str]) -> List[models.SparseVector]:
        embeddings = self._model.passage_embed(documents)
        return [
            models.SparseVector(**embedding.as_object()) for embedding in embeddings
        ]

    def embed_query(self, query: str) -> models.SparseVector:
        embedding = next(self._model.query_embed(query)).as_object()  # noqa
        return models.SparseVector(**embedding)

    def __str__(self):
        return f"SparseFastEmbedModelAdapter(model_name={self._model.model_name})"


class MultiVectorFastEmbedModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for the FastEmbed late interaction models.
    """

    def __init__(self, model_name: str):
        self._model = LateInteractionTextEmbedding(model_name=model_name)

    def embed_document(self, document: str) -> List[DenseVector]:
        return next(self._model.passage_embed([document])).tolist()  # noqa

    def embed_documents(self, documents: List[str]) -> List[List[DenseVector]]:
        embeddings = self._model.passage_embed(documents)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> List[DenseVector]:
        return next(self._model.query_embed(query)).tolist()  # noqa

    def __str__(self):
        return f"MultiVectorFastEmbedModelAdapter(model_name={self._model.model_name})"
