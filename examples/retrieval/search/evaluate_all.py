import logging

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.model_adapter.colbert import ColbertModelAdapter
from beir_qdrant.retrieval.model_adapter.fastembed import (
    DenseFastEmbedModelAdapter,
    MultiVectorFastEmbedModelAdapter,
    SparseFastEmbedModelAdapter,
)
from beir_qdrant.retrieval.model_adapter.sentence_transformers import (
    TokenEmbeddingsSentenceTransformerModelAdapter,
)
from beir_qdrant.retrieval.search.dense import DenseQdrantSearch
from beir_qdrant.retrieval.search.hybrid import RRFHybridQdrantSearch
from beir_qdrant.retrieval.search.multi_vector import MultiVectorQdrantSearch
from beir_qdrant.retrieval.search.sparse import SparseQdrantSearch

# Configure basic logging (level INFO)
logging.basicConfig(level=logging.INFO)

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
qdrant_client = QdrantClient("http://localhost:6333")

# Create all the searches to compare
base_searches = [
    DenseQdrantSearch(
        qdrant_client,
        model=DenseFastEmbedModelAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2",
        vector_name="all-MiniLM-L6-v2",
        initialize=True,
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        ),
    ),
    SparseQdrantSearch(
        qdrant_client,
        model=SparseFastEmbedModelAdapter(model_name="prithvida/Splade_PP_en_v1"),
        collection_name=f"{dataset}-splade",
        vector_name="splade",
        initialize=True,
    ),
    SparseQdrantSearch(
        qdrant_client,
        model=SparseFastEmbedModelAdapter("Qdrant/bm42-all-minilm-l6-v2-attentions"),
        collection_name=f"{dataset}-bm42",
        vector_name="bm42",
        initialize=True,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=MultiVectorFastEmbedModelAdapter(model_name="colbert-ir/colbertv2.0"),
        collection_name=f"{dataset}-colbert-fastembed",
        vector_name="colbert-fastembed",
        initialize=True,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=ColbertModelAdapter(
            model_name="jinaai/jina-colbert-v1-en", doc_maxlen=8192
        ),
        collection_name=f"{dataset}-jina-colbert",
        vector_name="jina-colbert",
        initialize=True,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=ColbertModelAdapter(model_name="colbert-ir/colbertv2.0"),
        collection_name=f"{dataset}-colbert-fastembed",
        vector_name="colbert",
        initialize=True,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=TokenEmbeddingsSentenceTransformerModelAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2-token-embeddings",
        vector_name="all-MiniLM-L6-v2-token-embeddings",
        initialize=True,
    ),
    MultiVectorQdrantSearch(
        qdrant_client,
        model=TokenEmbeddingsSentenceTransformerModelAdapter(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
        collection_name=f"{dataset}-all-MiniLM-L6-v2-token-embeddings-sq",
        vector_name="all-MiniLM-L6-v2-token-embeddings-sq",
        initialize=True,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8)
        ),
    ),
]

# Combine all the searches using Rank Fusion
rrf_search = RRFHybridQdrantSearch(
    qdrant_client,
    collection_name=f"{dataset}-rrf",
    initialize=True,
    searches=base_searches,
)
searches = base_searches + [rrf_search]

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
