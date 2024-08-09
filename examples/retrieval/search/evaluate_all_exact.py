import logging

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.models.colbert import ColbertModelAdapter
from beir_qdrant.retrieval.models.fastembed import SparseFastEmbedModelAdapter
from beir_qdrant.retrieval.models.sentence_transformers import (
    SentenceTransformerModelAdapter,
    TokenEmbeddingsSentenceTransformerModelAdapter,
)
from beir_qdrant.retrieval.search.dense import DenseQdrantSearch
from beir_qdrant.retrieval.search.multi_vector import MultiVectorQdrantSearch
from beir_qdrant.retrieval.search.sparse import SparseQdrantSearch

# Configure basic logging (level INFO)
logging.basicConfig(level=logging.WARNING)

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Download and load the dataset
dataset = "scifact"
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Connect to Qdrant running on localhost
qdrant_client = QdrantClient(
    "http://localhost:6333",
    timeout=60,
)

# Create all the searches to compare
searches = [
    SparseQdrantSearch(
        qdrant_client,
        model=SparseFastEmbedModelAdapter(model_name="prithvida/Splade_PP_en_v1"),
        collection_name=f"{dataset}-splade",
        vector_name="splade",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=128,
        model_batch_size=1024,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=ColbertModelAdapter(
            model_name="colbert-ir/colbertv2.0",
            query_maxlen=32,
            doc_maxlen=512,
        ),
        collection_name=f"{dataset}-colbert",
        vector_name="colbert",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=8,
        model_batch_size=1024,
    ),
    DenseQdrantSearch(
        qdrant_client,
        model=SentenceTransformerModelAdapter(
            model_path="sentence-transformers/all-MiniLM-L6-v2"
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2",
        vector_name="all-MiniLM-L6-v2",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=128,
        model_batch_size=1024,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=TokenEmbeddingsSentenceTransformerModelAdapter(
            model_path="sentence-transformers/all-MiniLM-L6-v2"
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2-token-embeddings",
        vector_name="all-MiniLM-L6-v2-token-embeddings",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=8,
        model_batch_size=1024,
    ),
    DenseQdrantSearch(
        qdrant_client,
        model=SentenceTransformerModelAdapter(model_path="BAAI/bge-small-en"),
        collection_name=f"{dataset}-BAAI-bge-small-en",
        vector_name="BAAI-bge-small-en",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=128,
        model_batch_size=1024,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=TokenEmbeddingsSentenceTransformerModelAdapter(
            model_path="BAAI/bge-small-en"
        ),
        collection_name=f"{dataset}-BAAI-bge-small-en-token-embeddings",
        vector_name="BAAI-bge-small-en-token-embeddings",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        batch_size=8,
        model_batch_size=1024,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=TokenEmbeddingsSentenceTransformerModelAdapter(
            model_path="sentence-transformers/all-MiniLM-L6-v2",
            precision="uint8",
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2-token-embeddings-uint8",
        vector_name="all-MiniLM-L6-v2-token-embeddings-uint8",
        initialize=True,
        clean_up=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000_000,
        ),
        search_params=models.SearchParams(
            exact=True,
        ),
        datatype=models.Datatype.UINT8,
        batch_size=16,
        model_batch_size=1024,
    ),
]

# Evaluate all the searches on the same test set
for model in searches:
    retriever = EvaluateRetrieval(model)
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )

    # Display the evaluation results
    print(f"Evaluation Results of model {model}:")
    print("NDCG@k:", ndcg)
    print("MAP@K:", _map)
    print("Recall@K:", recall)
    print("Precision@K:", precision)
