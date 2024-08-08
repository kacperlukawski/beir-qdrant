import itertools
from typing import Dict, Iterable, List, Union

import numpy as np
import scipy as sp
from fastembed import (
    LateInteractionTextEmbedding,
    SparseEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from torch import Tensor
from tqdm import tqdm

from beir_qdrant.retrieval.models.base import (
    BaseDenseModelAdapter,
    BaseMultiVectorModelAdapter,
    BaseSparseModelAdapter,
)


class DenseFastEmbedModelAdapter(BaseDenseModelAdapter):
    """
    Adapter for the FastEmbed text embedding models.
    """

    def __init__(self, model_name: str, sep: str = ""):
        super().__init__(sep=sep)

        try:
            import torch

            use_cuda = torch.cuda.is_available()
            providers = ["CUDAExecutionProvider"] if use_cuda else None
            self._model = TextEmbedding(model_name=model_name, providers=providers)
        except Exception:
            self._model = TextEmbedding(model_name=model_name)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        texts = self._format_corpus(corpus)
        embeddings = self._model.passage_embed(texts, batch_size=batch_size, **kwargs)
        return np.array(
            list(tqdm(embeddings, total=len(texts), desc="Encoding corpus"))
        )

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        embeddings = self._model.query_embed(queries, batch_size=batch_size, **kwargs)
        return np.array(
            list(tqdm(embeddings, total=len(queries), desc="Encoding queries"))
        )

    def __str__(self):
        return f"DenseFastEmbedModelAdapter(model_name={self._model.model_name})"


class SparseFastEmbedModelAdapter(BaseSparseModelAdapter):
    """
    Adapter for the FastEmbed sparse text embedding models.
    """

    def __init__(self, model_name: str, sep: str = " "):
        super().__init__(sep=sep)

        try:
            import torch

            use_cuda = torch.cuda.is_available()
            providers = ["CUDAExecutionProvider"] if use_cuda else None
            self._model = SparseTextEmbedding(
                model_name=model_name, providers=providers
            )
        except Exception:
            self._model = SparseTextEmbedding(model_name=model_name)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        texts = self._format_corpus(corpus)
        embeddings = self._model.passage_embed(texts, batch_size=batch_size, **kwargs)
        return self._to_sparse_matrix(tqdm(embeddings, total=len(texts), desc="Corpus"))

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        embeddings = self._model.query_embed(queries, batch_size=batch_size, **kwargs)
        return self._to_sparse_matrix(
            tqdm(embeddings, total=len(queries), desc="Queries")
        )

    def _to_sparse_matrix(self, embeddings: Iterable[SparseEmbedding]) -> np.ndarray:
        """
        Convert a list of sparse embeddings to a sparse matrix.
        :param embeddings:
        :return:
        """
        embeddings = list(embeddings)
        data = list(itertools.chain(*[embedding.values for embedding in embeddings]))
        rows = list(
            itertools.chain(
                *[[i] * len(embedding.values) for i, embedding in enumerate(embeddings)]
            )
        )
        cols = list(itertools.chain(*[embedding.indices for embedding in embeddings]))
        return sp.sparse.csr_array((data, (rows, cols)))

    def __str__(self):
        return f"SparseFastEmbedModelAdapter(model_name={self._model.model_name})"


class MultiVectorFastEmbedModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for the FastEmbed late interaction models.
    """

    def __init__(self, model_name: str, sep: str = " "):
        super().__init__(sep=sep)

        try:
            import torch

            use_cuda = torch.cuda.is_available()
            providers = ["CUDAExecutionProvider"] if use_cuda else None
            self._model = LateInteractionTextEmbedding(
                model_name=model_name, providers=providers
            )
        except Exception:
            self._model = LateInteractionTextEmbedding(model_name=model_name)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        texts = self._format_corpus(corpus)
        embeddings = self._model.passage_embed(texts, batch_size=batch_size, **kwargs)
        return [
            Tensor(embedding_list)
            for embedding_list in tqdm(
                embeddings, total=len(texts), desc="Encoding corpus"
            )
        ]

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        embeddings = self._model.query_embed(queries, batch_size=batch_size, **kwargs)
        return [
            Tensor(embedding_list)
            for embedding_list in tqdm(
                embeddings, total=len(queries), desc="Encoding queries"
            )
        ]

    def __str__(self):
        return f"MultiVectorFastEmbedModelAdapter(model_name={self._model.model_name})"
