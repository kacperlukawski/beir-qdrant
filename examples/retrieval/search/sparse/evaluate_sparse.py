import logging

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient

from beir_qdrant.retrieval.search.sparse import SparseQdrantSearch

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

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

# Create the retriever and evaluate it on the test set
model = SparseQdrantSearch(
    qdrant_client, collection_name="scifact-splade", initialize=True
)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

# Display the evaluation results
print("Evaluation Results:")
print("NDCG@k:", ndcg)
print("MAP@K:", _map)
print("Recall@K:", recall)
print("Precision@K:", precision)
