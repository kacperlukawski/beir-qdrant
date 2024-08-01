from typing import List, Literal, Optional

from sentence_transformers import SentenceTransformer

from beir_qdrant.retrieval.model_adapter.base import (
    BaseDenseModelAdapter,
    BaseMultiVectorModelAdapter,
    DenseVector,
)


class SentenceTransformerModelAdapter(BaseDenseModelAdapter):
    """
    Adapter for the SentenceTransformer models. They only provide dense embeddings, so it is not necessary to
    distinguish between dense and sparse models.
    """

    def __init__(
        self,
        model_name: str,
        document_prompt: Optional[str] = None,
        query_prompt: Optional[str] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    ):
        self._model = SentenceTransformer(model_name)
        self._document_prompt = document_prompt
        self._query_prompt = query_prompt
        self._precision = precision

    def embed_document(self, document: str) -> DenseVector:
        return self._model.encode(
            document, prompt=self._document_prompt, precision=self._precision
        ).tolist()

    def embed_query(self, query: str) -> DenseVector:
        return self._model.encode(
            query, prompt=self._query_prompt, precision=self._precision
        ).tolist()


class TokenEmbeddingsSentenceTransformerModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for any SentenceTransformer model that generates token embeddings. They might be used for multi-vector
    retrieval.
    """

    def __init__(
        self,
        model_name: str,
        document_prompt: Optional[str] = None,
        query_prompt: Optional[str] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    ):
        self._model = SentenceTransformer(model_name)
        self._document_prompt = document_prompt
        self._query_prompt = query_prompt
        self._precision = precision

    def embed_document(self, document: str) -> List[DenseVector]:
        return self._model.encode(
            document,
            prompt=self._document_prompt,
            precision=self._precision,
            output_value="token_embeddings",
        ).tolist()

    def embed_query(self, query: str) -> List[DenseVector]:
        return self._model.encode(
            query,
            prompt=self._query_prompt,
            precision=self._precision,
            output_value="token_embeddings",
        ).tolist()
