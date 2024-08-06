import abc
import logging
import time
import uuid
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from beir_qdrant.retrieval.model_adapter.base import BaseModelAdapter

logger = logging.getLogger(__name__)


def batched(iterable, n=1):
    """
    Batch an iterable into chunks of size n.
    :param iterable:
    :param n:
    :return:
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    payload: Dict[str, str]


class QdrantBase(abc.ABC):
    """
    Base class for Qdrant based search.
    """

    SLEEP_INTERVAL = 1.0

    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 64,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.initialize = initialize
        self.clean_up = clean_up
        self.optimizers_config = optimizers_config
        self.batch_size = batch_size

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

        if self.clean_up:
            self.qdrant_client.delete_collection(self.collection_name)

        return results

    def index_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """
        Store the corpus in the Qdrant collection.
        :param corpus:
        :return:
        """
        # Iterate corpus in batches, not one by one
        corpus_items = corpus.items()
        for batch in batched(
            tqdm(corpus_items, desc="Corpus indexing"), self.batch_size
        ):
            documents = [Document(doc_id, doc) for doc_id, doc in batch]

            init_time = time.perf_counter()
            points = self.docs_to_points(documents)
            end_time = time.perf_counter()
            logger.info(
                f"Converted {len(points)} documents to points in {end_time - init_time:.8f} seconds"
            )

            init_time = time.perf_counter()
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points,
            )
            end_time = time.perf_counter()
            logger.info(
                f"Uploaded {len(points)} points in {end_time - init_time:.8f} seconds"
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
    def docs_to_points(self, documents: List[Document]) -> List[models.PointStruct]:
        """
        Convert a list of documents to Qdrant points.
        :param documents:
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
        init_time = time.perf_counter()
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        while collection_info.status != models.CollectionStatus.GREEN:
            logger.info(f"Collection status: {collection_info.status}")
            time.sleep(self.SLEEP_INTERVAL)
            collection_info = self.qdrant_client.get_collection(self.collection_name)
        end_time = time.perf_counter()
        logger.info(f"Collection indexed in {end_time - init_time:.8f} seconds")

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(self._str_params())})"

    def _str_params(self) -> List[str]:
        return [
            f"collection_name={self.collection_name}",
            f"initialize={self.initialize}",
            f"clean_up={self.clean_up}",
            f"optimizers_config={self.optimizers_config}",
        ]


class SingleNamedVectorQdrantBase(QdrantBase, abc.ABC):
    """
    Base class for Qdrant based search with a single vector. Single vector means here, a single named vector of a point.
    Even if it contains multiple token-level embeddings, we still think of it as a single vector. Contrast this with
    hybrid search, where we have multiple named vectors used downstream.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        model: BaseModelAdapter,
        collection_name: str,
        initialize: bool = True,
        clean_up: bool = False,
        optimizers_config: Optional[models.OptimizersConfigDiff] = None,
        batch_size: int = 64,
        vector_name: str = "vector",
        search_params: Optional[models.SearchParams] = None,
    ):
        super().__init__(
            qdrant_client,
            collection_name,
            initialize,
            clean_up,
            optimizers_config,
            batch_size,
        )
        self.model = model
        self.vector_name = vector_name
        self.search_params = search_params

    def handle_query(self, query: str, limit: int) -> List[models.ScoredPoint]:
        query_embedding = self.model.embed_query(query)
        init_time = time.perf_counter()
        result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=self.search_params,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Queried {self.collection_name} in {end_time - init_time:.8f} seconds"
        )

        return result.points

    def docs_to_points(self, documents: List[Document]) -> List[models.PointStruct]:
        # Create embeddings for all the documents in a single model call
        texts = [doc.payload["text"] for doc in documents]
        embeddings = self.model.embed_documents(texts)

        points = []
        for doc, embedding in zip(documents, embeddings):
            points.append(
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={self.vector_name: embedding},
                    payload={"doc_id": doc.id, **doc.payload},
                )
            )
        return points

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"model={self.model}",
            f"vector_name={self.vector_name}",
            f"search_params={self.search_params}",
        ]
