import time

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient, models

from beir_qdrant.retrieval.models.sentence_transformers import (
    SentenceTransformerModelAdapter,
)
from beir_qdrant.retrieval.search.dense import DenseQdrantSearch

client = QdrantClient(location="http://localhost:6333")

datasets = [
    "scifact",
    "trec-covid",
    "arguana",
    "nfcorpus",
]

# Download and load the dataset
for dataset in datasets:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
    data_path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # Measure the time taken for indexing and retrieval
    start_time = time.monotonic()

    # Define all the models and hyperparameters
    test_models = [
        # Standard retrieval model
        DenseQdrantSearch(
            client,
            model=SentenceTransformerModelAdapter(
                "sentence-transformers/static-retrieval-mrl-en-v1"
            ),
            # Ensure HNSW index will be used for search
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=1,
            ),
            collection_name="static-retrieval-mrl-en-v1",
            vector_name="dense",
            initialize=True,
        ),
        # Binary quantization enabled
        DenseQdrantSearch(
            client,
            model=SentenceTransformerModelAdapter(
                "sentence-transformers/static-retrieval-mrl-en-v1"
            ),
            # Ensure HNSW index will be used for search
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=1,
            ),
            collection_name="static-retrieval-mrl-en-v1-bq",
            vector_name="dense",
            initialize=True,
            # Enable Binary Quantization
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,
                )
            ),
            # Disable rescoring, to use just the binary vectors
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=False,
                )
            ),
        ),
    ]

    # Test the static embedding model on a selected dataset
    print(f"Dataset used: {dataset}")
    for model in test_models:
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)

        # Log the time taken for indexing and retrieval
        end_time = time.monotonic()
        duration = end_time - start_time
        print(f" Model: {model}")
        print(f" Time taken for indexing and retrieval: {duration:.2f} seconds")

        # Display the evaluation results
        ndcg, _map, recall, precision = retriever.evaluate(
            qrels, results, retriever.k_values
        )
        print(" Evaluation results:")
        print("  NDCG@k:", ndcg)
        print("  MAP@K:", _map)
        print("  Recall@K:", recall)
        print("  Precision@K:", precision)
