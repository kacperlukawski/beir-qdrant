# beir-qdrant

[BEIR](https://github.com/beir-cellar/beir) is a heterogeneous benchmark containing diverse IR tasks. This project
integrates BEIR with [Qdrant](https://github.com/qdrant/qdrant) - a vector search engine.

## Installation

```bash
pip install beir-qdrant
```

## Quick Example

The following example demonstrates how to use BEIR with Qdrant dense search. The example uses the SciFact dataset
and `all-MiniLM-L6-v2` model from Sentence Transformers to generate the dense embeddings.

```python
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from qdrant_client import QdrantClient

from beir_qdrant.retrieval.search.dense import DenseQdrantSearch

# Download and load the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Connect to Qdrant running on localhost
qdrant_client = QdrantClient("http://localhost:6333")

# Create the retriever and evaluate it on the test set
model = DenseQdrantSearch(
    qdrant_client,
    collection_name="scifact-all-MiniLM-L6-v2",
    dense_model_name="sentence-transformers/all-MiniLM-L6-v2",
    initialize=True,
)
retriever = EvaluateRetrieval(model)
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

The example above demonstrates how to use the dense embeddings, but changing the search mode is as simple as changing
the model implementation.

## Supported Modes

Qdrant supports different search modes, including:

- Dense search: `beir_qdrant.retrieval.search.dense.DenseQdrantSearch`
- Sparse search: `beir_qdrant.retrieval.search.sparse.SparseQdrantSearch`
- BM42 search: `beir_qdrant.retrieval.search.sparse.BM42Search`
- Late interaction: `beir_qdrant.retrieval.search.late_interaction.LateInteractionQdrantSearch`
- Hybrid search with RRF: `beir_qdrant.retrieval.search.hybrid.RRFHybridQdrantSearch`
