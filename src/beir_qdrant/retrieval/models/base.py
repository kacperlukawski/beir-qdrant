import abc
from typing import Dict, List, Union

import numpy as np
from torch import Tensor


class BaseModelAdapter(abc.ABC):
    """
    This package does not assume anything about the models used to generate embeddings. However, it has to use the same
    interface for all models, so that they can be used interchangeably.

    Adapter pattern is used to adapt different models to the same interface. This is the base class for all model
    adapters, no matter the type of the model and the output it generates.
    """

    def __init__(self, sep: str = " "):
        """
        :param sep: A separator between different data fields
        """
        self.sep = sep

    @abc.abstractmethod
    def encode_corpus(
        self,
        corpus: Union[List[Dict[str, str]], Dict[str, List]],
        batch_size: int = 8,
        **kwargs,
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def encode_queries(
        self, queries: List[str], batch_size: int = 16, **kwargs
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        raise NotImplementedError

    def _format_corpus(
        self, corpus: Union[List[Dict[str, str]], Dict[str, List]]
    ) -> List[str]:
        # TODO: check the logic and simplify, as this is derived from BeIR
        if type(corpus) is dict:
            return [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        return [
            (doc["title"] + self.sep + doc["text"]).strip()
            if "title" in doc
            else doc["text"].strip()
            for doc in corpus
        ]


class BaseDenseModelAdapter(BaseModelAdapter, abc.ABC):
    """
    Adapter for all the dense embedding models. Dense models generate a single vector for each document or query.
    """

    pass


class BaseSparseModelAdapter(BaseModelAdapter, abc.ABC):
    """
    Adapter for all the sparse embedding models. Sparse models generate a single vector for document or query, but the
    majority of the values are zeros. Practically, this means that the vectors are stored in a sparse format as list
    of non-zero values and their indices. Qdrant has a corresponding sparse vector type, so it is also used here.
    """

    pass


class BaseMultiVectorModelAdapter(BaseModelAdapter, abc.ABC):
    """
    Adapter for models that generate multiple vectors for each document or query. This is used for models that generate
    token-level embeddings, such as late interaction models, including ColBERT.
    """

    pass
