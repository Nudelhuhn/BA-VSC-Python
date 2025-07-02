import time
complete_time = time.time()
start = time.time()
import numpy as np
import yaml
import warnings
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
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


    # Embedding
    print("⏳ Embeddings werden berechnet...")
    start = time.time()
    model = EmbeddingModel(config['embedding']['model'])
    embeddings = np.array([model.get_embedding(code) for code in code_snippets])    # np.array needed for .shape method of pca and in general more universal than normal lists
    if len(embeddings) - 1 <= 2:                                                    # more than three files have to be embedded
        print("please choose at least four files")
        return
    print(f"embedding_model {time.time() - start:.2f} Sekunden")


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


    # # interactive plotting (show file name by hovering)
    # start = time.time()
    # AdvancedInteractivePlot.ad_int_plot(reduced_embeddings, labels, loader.get_filenames(), loader.get_parent_dirs())
    # print(f"interactive_plot {time.time() - start:.2f} Sekunden")


    # Visualization
    start = time.time()
    ClusterPlotter.plot(reduced_embeddings, labels)
    print(f"Visualization {time.time() - start:.2f} Sekunden")
    

    # Evaluation
    start = time.time()
    results = EvaluationMetrics.evaluate(reduced_embeddings, labels)
    print("Evaluation results:", results)
    print(f"Evaluation {time.time() - start:.2f} Sekunden")

    print(f"Komplett {time.time() - complete_time} Sekunden")

if __name__ == "__main__":
    run_pipeline()