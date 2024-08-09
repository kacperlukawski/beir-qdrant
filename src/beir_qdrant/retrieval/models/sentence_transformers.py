from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from beir.retrieval.models import SentenceBERT
from torch import Tensor

from beir_qdrant.retrieval.models.base import (
    BaseDenseModelAdapter,
    BaseMultiVectorModelAdapter,
)


class SentenceTransformerModelAdapter(SentenceBERT, BaseDenseModelAdapter):
    """
    Adapter for the SentenceTransformer models providing dense text embeddings.
    """

    def __init__(
        self,
        model_path: Union[str, Tuple],
        document_prompt: Optional[str] = None,
        query_prompt: Optional[str] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        sep: str = " ",
    ):
        super().__init__(model_path, sep=sep)
        self._document_prompt = document_prompt
        self._query_prompt = query_prompt
        self._precision = precision

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return super().encode_queries(
            queries,
            batch_size,
            prompt=self._query_prompt,
            precision=self._precision,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return super().encode_corpus(
            corpus,
            batch_size,
            prompt=self._document_prompt,
            precision=self._precision,
            **kwargs,
        )

    def __str__(self):
        return (
            f"SentenceTransformerModelAdapter("
            f"model_name={self.q_model.model_card_data.base_model}, "
            f"document_prompt={self._document_prompt}, "
            f"query_prompt={self._query_prompt}, "
            f"precision={self._precision}"
            f")"
        )


class TokenEmbeddingsSentenceTransformerModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for any SentenceTransformer model that generates token embeddings. They might be used for multi-vector
    retrieval.
    """

    def __init__(
        self,
        model_path: Union[str, Tuple],
        document_prompt: Optional[str] = None,
        query_prompt: Optional[str] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        sep: str = " ",
    ):
        super().__init__(sep=sep)
        self._model = SentenceTransformerModelAdapter(
            model_path, document_prompt, query_prompt, precision, sep
        )

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self._model.encode_corpus(
            corpus, batch_size, output_value="token_embeddings", **kwargs
        )

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self._model.encode_queries(
            queries, batch_size, output_value="token_embeddings", **kwargs
        )

    def __str__(self):
        return (
            f"TokenEmbeddingsSentenceTransformerModelAdapter("
            f"_model={self._model}"
            f")"
        )
