import logging

from qdrant_client import QdrantClient

from beir_qdrant.retrieval.search.sparse.sparse_search import SparseQdrantSearch

logger = logging.getLogger(__name__)


class BM42Search(SparseQdrantSearch):
    """
    BM42 is a retrieval model that uses the attention values of the tokens in the document to represent the importance
    of the token in the document or query.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        vector_name: str = "bm42",
        attention_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
    ):
        super().__init__(
            qdrant_client,
            collection_name,
            initialize,
            vector_name,
            attention_model_name,
        )
