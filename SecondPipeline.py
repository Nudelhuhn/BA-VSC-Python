import time
complete_time = time.time()
start = time.time()
import os
import numpy as np
import yaml
import warnings
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimension_reducer import DimensionalityReducer
from clustering.clustering_engine import ClusteringEngine
from visualization.advanced_interactive_plot import AdvancedInteractivePlot
from visualization.cluster_plotter import ClusterPlotter
from evaluation.evaluation_metrics import EvaluationMetrics
print(f"imports {time.time() - start:.2f} Sekunden")


start = time.time()
def load_config(path="config.yaml"):
    with open(path, "r") as file:               # open and read (r) file
        return yaml.safe_load(file)             # load file
print(f"config {time.time() - start:.2f} Sekunden")


def run_pipeline():
    config = load_config()


    # ignore some unimportant warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # ignores the warning "FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8. warnings.warn("
    warnings.filterwarnings("ignore", category=UserWarning)     # ignores the warning "UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism. warn("

    # load data
    start = time.time()
    loader = DataLoader(config['data']['files_to_look_for'], config['data']['input_path'], config['data']['exclude_files'])
    code_snippets = loader.load_code_files()
    print(f"data_loader {time.time() - start:.2f} Sekunden")


    # # Embedding
    # print("⏳ Embeddings werden berechnet...")
    # start = time.time()
    # model = EmbeddingModel(config['embedding']['model'])
    # embeddings = np.array([model.get_embedding(code) for code in code_snippets])    # np.array needed for .shape method of pca and in general more universal than normal lists
    # if len(embeddings) - 1 <= 2:                                                    # more than three files have to be embedded
    #     print("please choose at least four files")
    #     return
    # print(f"embedding_model {time.time() - start:.2f} Sekunden")

    # Caching: existiert eine Cache-Datei?
    model = EmbeddingModel(config['embedding']['model'])
    cache_path = "cached_embeddings.npy"

    # old caching
    # if os.path.exists(cache_path):
    #     embeddings = np.load(cache_path)
    #     print("✅ Embeddings aus Cache geladen.")
    # else:
    #     # Embeddings berechnen (Batch-Verarbeitung)
    #     print("⏳ Embeddings werden berechnet...")
    #     start = time.time()
    #     embeddings = np.array([model.get_embedding(code) for code in code_snippets])
    #     end = time.time()
    #     print(f"✅ Embedding-Dauer (Batch): {end - start:.2f} Sekunden")

    #     # In Cache speichern
    #     np.save(cache_path, embeddings)
    #     print("💾 Embeddings im Cache gespeichert.")

    # new caching
    if os.path.exists(cache_path):
        embeddings_cache = np.load(cache_path)
        if len(embeddings_cache) >= len(code_snippets):
            # Nur die benötigten Embeddings laden
            embeddings = embeddings_cache[:len(code_snippets)]
            print(f"✅ {len(code_snippets)} Embeddings aus Cache geladen.")
        else:
            print("⚠️ Nicht genügend gecachte Embeddings vorhanden. Berechne neu...")
            embeddings = np.array([model.get_embedding(code) for code in code_snippets])
            np.save(cache_path, embeddings)
            print("💾 Embeddings im Cache gespeichert.")
    else:
        print("⏳ Embeddings werden berechnet...")
        embeddings = np.array([model.get_embedding(code) for code in code_snippets])
        np.save(cache_path, embeddings)
        print("💾 Embeddings im Cache gespeichert.")



    # Dimensionality reduction
    start = time.time()
    reducer = DimensionalityReducer(config['dim_reduction']['method'], config['dim_reduction']['params'])
    reduced_embeddings = reducer.reduce(embeddings)
    print(f"Reducer {time.time() - start:.2f} Sekunden")


    # Clustering on the reduced Embeddings
    start = time.time()
    clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
    labels = clusterer.cluster(reduced_embeddings)  # colorbar values of diagram and in general the clustering allocations
    print(f"Clustering {time.time() - start:.2f} Sekunden")


    # interactive plotting (show file name by hovering)
    start = time.time()
    AdvancedInteractivePlot.ad_int_plot(reduced_embeddings, labels, loader.get_filenames())
    print(f"interactive_plot {time.time() - start:.2f} Sekunden")


    # # Visualization
    # start = time.time()
    # ClusterPlotter.plot(reduced_embeddings, labels)
    # print(f"Visualization {time.time() - start:.2f} Sekunden")
    

    # Evaluation
    start = time.time()
    results = EvaluationMetrics.evaluate(reduced_embeddings, labels)
    print("Evaluation results:", results)
    print(f"Evaluation {time.time() - start:.2f} Sekunden")

    print(f"Komplett {time.time() - complete_time} Sekunden")

if __name__ == "__main__":
    run_pipeline()