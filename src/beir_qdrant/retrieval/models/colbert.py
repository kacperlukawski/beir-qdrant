from typing import Dict, List, Union

import numpy as np
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from torch import Tensor
from tqdm import tqdm

from beir_qdrant.retrieval.models.base import BaseMultiVectorModelAdapter


class ColbertModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for the Colbert model. It provides multimple dense embeddings for both documents and queries.
    """

    def __init__(
        self,
        model_name: str,
        query_maxlen: int = 128,
        doc_maxlen: int = 512,
        sep: str = " ",
    ):
        super().__init__(sep=sep)
        self._model_name = model_name
        self._colbert_config = ColBERTConfig(
            query_maxlen=query_maxlen, doc_maxlen=doc_maxlen
        )
        self._checkpoint = Checkpoint(model_name, colbert_config=self._colbert_config)

    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        texts = self._format_corpus(corpus)
        embeddings = self._checkpoint.docFromText(texts, bsize=batch_size, **kwargs)
        return [
            Tensor(embedding_list)
            for embedding_list in tqdm(
                embeddings, total=len(texts), desc="Encoding corpus"
            )
        ]

    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        embeddings = self._checkpoint.queryFromText(queries, bsize=batch_size, **kwargs)
        return [
            Tensor(embedding_list)
            for embedding_list in tqdm(
                embeddings, total=len(queries), desc="Encoding queries"
            )
        ]

    def __str__(self):
        return (
            f"ColbertModelAdapter(model_name={self._model_name}, "
            f"query_maxlen={self._colbert_config.query_maxlen}, "
            f"doc_maxlen={self._colbert_config.doc_maxlen})"
        )
