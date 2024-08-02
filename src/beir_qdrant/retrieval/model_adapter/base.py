import abc
from typing import List, Union

from qdrant_client import models

DenseVector = List[float]
AnyVector = Union[DenseVector, models.SparseVector, List[DenseVector]]


class BaseModelAdapter(abc.ABC):
    """
    This package does not assume anything about the models used to generate embeddings. However, it has to use the same
    interface for all models, so that they can be used interchangeably.

    Adapter pattern is used to adapt different models to the same interface. This is the base class for all model
    adapters, no matter the type of the model and the output it generates.
    """

    @abc.abstractmethod
    def embed_documents(self, documents: List[str]) -> List[AnyVector]:
        """
        Embed multiple documents.
        :param documents:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, query: str) -> AnyVector:
        raise NotImplementedError


class BaseDenseModelAdapter(abc.ABC):
    """
    Adapter for all the dense embedding models. Dense models generate a single vector for each document or query.
    """

    @abc.abstractmethod
    def embed_documents(self, documents: List[str]) -> List[DenseVector]:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, query: str) -> DenseVector:
        raise NotImplementedError


class BaseSparseModelAdapter(abc.ABC):
    """
    Adapter for all the sparse embedding models. Sparse models generate a single vector for document or query, but the
    majority of the values are zeros. Practically, this means that the vectors are stored in a sparse format as list
    of non-zero values and their indices. Qdrant has a corresponding sparse vector type, so it is also used here.
    """

    @abc.abstractmethod
    def embed_documents(self, documents: List[str]) -> List[models.SparseVector]:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, query: str) -> models.SparseVector:
        raise NotImplementedError


class BaseMultiVectorModelAdapter(abc.ABC):
    """
    Adapter for models that generate multiple vectors for each document or query. This is used for models that generate
    token-level embeddings, such as late interaction models, including ColBERT.
    """

    @abc.abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[DenseVector]]:
        raise NotImplementedError

    @abc.abstractmethod
    def embed_query(self, query: str) -> List[DenseVector]:
        raise NotImplementedError
