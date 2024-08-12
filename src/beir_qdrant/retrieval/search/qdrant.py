import abc
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient, models
from tqdm import tqdm

from beir_qdrant.retrieval.models.base import BaseModelAdapter

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
        model_batch_size: Optional[int] = None,
    ):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.initialize = initialize
        self.clean_up = clean_up
        self.optimizers_config = optimizers_config
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size or batch_size

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
        results = self.handle_queries(queries, top_k)

        if self.clean_up:
            self.qdrant_client.delete_collection(self.collection_name)

        return results

    def index_corpus(self, corpus: Dict[str, Dict[str, str]]):
        """
        Store the corpus in the Qdrant collection.
        :param corpus:
        :return:
        """
        points = self.corpus_to_points(corpus)
        init_time = time.perf_counter()
        self.qdrant_client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=self.batch_size,
        )
        end_time = time.perf_counter()
        logger.info(f"Uploaded points in {end_time - init_time:.8f} seconds")

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
    def corpus_to_points(
        self, corpus: Dict[str, Dict[str, str]]
    ) -> Iterable[models.PointStruct]:
        """
        Convert a corpus to Qdrant points.
        :param corpus:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def handle_queries(
        self, queries: Dict[str, str], limit: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Handle all the queries in the given iterable.
        :param queries:
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
        model_batch_size: Optional[int] = None,
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
            model_batch_size,
        )
        self.model = model
        self.vector_name = vector_name
        self.search_params = search_params

    def handle_queries(
        self, queries: Dict[str, str], limit: int
    ) -> Dict[str, Dict[str, float]]:
        query_ids, query_texts = zip(*queries.items())

        init_time = time.perf_counter()
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.model_batch_size
        )
        end_time = time.perf_counter()
        logger.info(
            f"Encoded {len(query_ids)} queries in {end_time - init_time:.8f} seconds"
        )

        # Convert the embeddings to the Qdrant format
        query_embeddings = self.convert_embeddings_to_qdrant_format(query_embeddings)

        results = defaultdict(dict)
        init_time = time.perf_counter()
        for query_id, query_embedding in tqdm(
            zip(query_ids, query_embeddings), total=len(query_ids), desc="Queries"
        ):
            query_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=self.vector_name,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                search_params=self.search_params,
            )
            results[query_id] = {
                point.payload["doc_id"]: point.score for point in query_results.points
            }
        end_time = time.perf_counter()
        logger.info(
            f"Queried {self.collection_name} in {end_time - init_time:.8f} seconds"
        )

        return results

    def corpus_to_points(
        self, corpus: Dict[str, Dict[str, str]]
    ) -> Iterable[models.PointStruct]:
        corpus_items = corpus.items()

        num_total_batches = len(corpus_items) // self.model_batch_size + 1
        for corpus_batch in tqdm(
            batched(corpus_items, self.model_batch_size),
            total=num_total_batches,
            desc="Corpus batches",
        ):
            document_ids, documents = zip(*corpus_batch)

            embeddings = self.model.encode_corpus(
                documents, batch_size=self.model_batch_size
            )

            # Convert the embeddings to the Qdrant format
            embeddings = self.convert_embeddings_to_qdrant_format(embeddings)

            for document_idx, (document_id, document_embedding) in enumerate(
                zip(document_ids, embeddings)
            ):
                yield models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector={self.vector_name: document_embedding},
                    payload={"doc_id": document_id, **documents[document_idx]},
                )

    def convert_embeddings_to_qdrant_format(self, embeddings):
        """
        Convert the query embeddings to the Qdrant format. This is a no-op for single vector searches.
        Some of the methods in the derived classes may require this conversion.
        :param embeddings:
        :return:
        """
        return embeddings

    def _str_params(self) -> List[str]:
        return super()._str_params() + [
            f"model={self.model}",
            f"vector_name={self.vector_name}",
            f"search_params={self.search_params}",
        ]
