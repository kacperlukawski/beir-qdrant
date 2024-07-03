import abc
import logging
import time
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models
from tqdm import tqdm

logger = logging.getLogger(__name__)


class QdrantBase(abc.ABC):
    """
    Base class for Qdrant based search.
    """

    SLEEP_INTERVAL = 1.0
    BATCH_SIZE = 64

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.initialize = initialize

    def search(
        self,
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if self.initialize:
            self.recreate_collection()
            self.index_corpus(corpus)
            self.await_indexing()

        # Run all the queries and collect the results
        results = {}
        for query_id, query in tqdm(queries.items(), desc="Queries"):
            points = self.handle_query(query, top_k)
            results[query_id] = {
                point.payload["doc_id"]: point.score for point in points
            }

        return results

    def index_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """
        Store the corpus in the Qdrant collection.
        :param corpus:
        :return:
        """
        points = []
        for doc_id, doc in tqdm(corpus.items(), desc="Corpus indexing"):
            points.append(self.doc_to_point(doc_id, doc))

        self.qdrant_client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.BATCH_SIZE,
        )

    def recreate_collection(self):
        """
        Create a collection in Qdrant. If it already exists, delete it and create a new one.
        :return:
        """
        collection_config = self.collection_config()
        self.qdrant_client.delete_collection(self.collection_name)
        self.qdrant_client.create_collection(**collection_config)

    @abc.abstractmethod
    def collection_config(self) -> Dict[str, Any]:
        """
        Return the configuration for the collection.
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_point(self, doc_id: str, doc: Dict[str, str]) -> models.PointStruct:
        """
        Convert a document to a Qdrant point.
        :param doc_id:
        :param doc:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        """
        Handle the query by searching the collection.
        :param query:
        :param limit:
        :return:
        """
        raise NotImplementedError

    def await_indexing(self):
        """
        Wait for the collection to be indexed.
        :return:
        """
        time.sleep(self.SLEEP_INTERVAL)
        logger.info("Waiting for collection to be indexed")
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        while collection_info.status != models.CollectionStatus.GREEN:
            logger.info(f"Collection status: {collection_info.status}")
            time.sleep(self.SLEEP_INTERVAL)
            collection_info = self.qdrant_client.get_collection(self.collection_name)

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(self._str_params())})"

    def _str_params(self) -> List[str]:
        return [
            f"collection_name={self.collection_name}",
        ]


class SingleVectorQdrantBase(QdrantBase):
    """
    Base class for Qdrant based search with a single vector.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        vector_name: str = "vector",
    ):
        super().__init__(qdrant_client, collection_name, initialize)
        self.vector_name = vector_name

    @abc.abstractmethod
    def create_document_vector(self, document: str) -> models.Vector:
        raise NotImplementedError

    @abc.abstractmethod
    def create_query_vector(self, query: str) -> models.Vector:
        raise NotImplementedError
