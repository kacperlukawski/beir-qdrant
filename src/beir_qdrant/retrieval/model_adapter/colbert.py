from typing import List

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from beir_qdrant.retrieval.model_adapter.base import (
    BaseMultiVectorModelAdapter,
    DenseVector,
)


class ColbertModelAdapter(BaseMultiVectorModelAdapter):
    """
    Adapter for the Colbert model. It provides multimple dense embeddings for both documents and queries.
    """

    def __init__(self, model_name: str, query_maxlen: int = 128, doc_maxlen: int = 512):
        self._model_name = model_name
        self._colbert_config = ColBERTConfig(
            query_maxlen=query_maxlen, doc_maxlen=doc_maxlen
        )
        self._checkpoint = Checkpoint(model_name, colbert_config=self._colbert_config)

    def embed_documents(self, documents: List[str]) -> List[List[DenseVector]]:
        embeddings = self._checkpoint.docFromText(documents)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, query: str) -> List[DenseVector]:
        embeddings = self._checkpoint.queryFromText([query])
        return embeddings[0].tolist()

    def __str__(self):
        return (
            f"ColbertModelAdapter(model_name={self._model_name}, "
            f"query_maxlen={self._colbert_config.query_maxlen}, "
            f"doc_maxlen={self._colbert_config.doc_maxlen})"
        )
