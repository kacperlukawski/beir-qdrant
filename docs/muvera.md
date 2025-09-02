---
title: Muvera Dual Vector Search
---

# Muvera Dual Vector Search

Muvera is an advanced dual-vector search approach that combines the benefits of dense and multivector embeddings for
improved retrieval performance. It uses Qdrant's multi-stage search capabilities with prefetch to achieve efficient and
accurate results.

## How It Works

The Muvera approach employs a two-stage retrieval process:

1. **Initial Retrieval**: Uses Muvera postprocessor to create dense vectors from multivector embeddings
2. **Final Reranking**: Applies the original multivector embeddings for precise scoring of the retrieved candidates

This dual approach leverages the speed of dense vector search for initial candidate selection while maintaining the
accuracy of multivector representations for final ranking.

## Key Components

- **`MuveraReranking`**: The main search class that orchestrates the dual-vector retrieval
- **`MuveraPostprocessorAdapter`**: Transforms multivector embeddings into dense representations using configurable dimensionality reduction parameters
- **`MultiVectorFastEmbedModelAdapter`**: Handles the original multivector embeddings for reranking

## Usage Example

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient

from beir_qdrant.retrieval.search.multi_vector.muvera_reranking import MuveraReranking

# Download and load the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Connect to Qdrant running on localhost
qdrant_client = QdrantClient("http://localhost:6333")

# Create the Muvera retriever
model = MuveraReranking(
    qdrant_client=qdrant_client,
    multivector_model_name="colbert-ir/colbertv2.0",  # Base multivector model
    collection_name="scifact-muvera",
    oversample_factor=5,  # Retrieve 5x more candidates for reranking
    initialize=True,
    # Muvera postprocessor parameters
    k_sim=4,          # Number of similar vectors for approximation
    dim_proj=32,      # Projected dimension size
    r_reps=10,        # Number of repetitions for stability
    random_seed=42,   # Random seed for reproducibility
)

# Evaluate the retriever
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```
