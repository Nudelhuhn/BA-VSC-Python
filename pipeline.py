#debugging
import time
import os
import numpy as np
#debugging
import yaml
start = time.time()
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
from clustering.clustering_engine import ClusteringEngine
from evaluation.evaluation_metrics import EvaluationMetrics
from visualization.cluster_plotter import ClusterPlotter
print(f"imports {time.time() - start:.2f} Sekunden")

start = time.time()
def load_config(path="config.yaml"):
    with open(path, "r") as file:               # open and read (r) file
        return yaml.safe_load(file)             # load file
end = time.time()
print(f"config {end - start:.2f} Sekunden")

def run_pipeline():
    config = load_config()

    # Daten laden
    start = time.time()
    loader = DataLoader(config['data']['input_path'])
    code_snippets = loader.load_code_files()
    end = time.time()
    print(f"data_loader {end - start:.2f} Sekunden")

    # Embeddings erzeugen
    model = EmbeddingModel(config['embedding']['model'])
    start = time.time()
    embeddings = [model.get_embedding(code) for code in code_snippets]
    end = time.time()
    print(f"embedding_model {end - start:.2f} Sekunden")

    # Caching: existiert eine Cache-Datei?
    cache_path = "cached_embeddings.npy"
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print("✅ Embeddings aus Cache geladen.")
    else:
        # Embeddings berechnen (Batch-Verarbeitung)
        # print("⏳ Embeddings werden berechnet...")
        start = time.time()
        embeddings = model.get_embedding(code_snippets)
        end = time.time()
        print(f"✅ Embedding-Dauer (Batch): {end - start:.2f} Sekunden")

        # In Cache speichern
        np.save(cache_path, embeddings)
        # print("💾 Embeddings im Cache gespeichert.")

    # Dimensionality Reduction
    # embeddings = np.random.rand(16, 768)
    start = time.time()
    reducer = DimensionalityReducer(method="umap", params={"n_components": config['dim_reduction']['n_components']})
    # print("Shape der Embeddings vor der Dimensionsreduktion:", embeddings.shape)
    reduced_embeddings = reducer.reduce(embeddings)
    # print("Shape der Embeddings nach der Dimensionsreduktion:", reduced_embeddings.shape)
    print(f"reducer {time.time() - start:.2f} Sekunden")

    # Clustering auf den reduzierten Embeddings
    start = time.time()
    clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
    labels = clusterer.cluster(reduced_embeddings)
    unique_labels, counts = np.unique(labels, return_counts=True)
    # print("Cluster-Labels:", labels)
    # print("Anzahl Cluster:", len(set(labels)) - (1 if -1 in labels else 0))
    # print("Label-Verteilung:", dict(zip(unique_labels, counts)))
    print(f"clustering {time.time() - start:.2f} Sekunden")

    # Visualisierung
    start = time.time()
    plotter = ClusterPlotter()
    plotter.plot(reduced_embeddings, labels)
    print(f"visualization {time.time() - start:.2f} Sekunden")
    
    # # Evaluation
    # results = EvaluationMetrics.evaluate(embeddings, labels)
    # print("Evaluationsergebnisse:", results)

if __name__ == "__main__":
    run_pipeline()
