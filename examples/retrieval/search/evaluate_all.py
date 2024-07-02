from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient

from beir_qdrant.retrieval.search.dense import DenseQdrantSearch
from beir_qdrant.retrieval.search.late_interaction import LateInteractionQdrantSearch
from beir_qdrant.retrieval.search.sparse import BM42Search, SparseQdrantSearch

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

# Create all the models to compare
models = [
    DenseQdrantSearch(
        qdrant_client,
        collection_name="scifact-all-MiniLM-L6-v2",
        dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
    SparseQdrantSearch(
        qdrant_client,
        collection_name="scifact-splade",
        sparse_model_name="prithvida/Splade_PP_en_v1",
    ),
    BM42Search(
        qdrant_client,
        collection_name="scifact-bm42",
        attention_model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
    ),
    LateInteractionQdrantSearch(
        qdrant_client,
        collection_name="scifact-colbert",
        late_interaction_model_name="colbert-ir/colbertv2.0",
    ),
]

# Evaluate all the models on the same test set
for model in models:
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
