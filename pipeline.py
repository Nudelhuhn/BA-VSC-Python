#debugging
import time
start = time.time()
#debugging
import yaml
import warnings
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
from clustering.clustering_engine import ClusteringEngine
from interactivePlot.interactive_plot import InteractivePlot
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
    loader = DataLoader(config['data']['file_name'], config['data']['input_path'])
    code_snippets = loader.load_code_files()
    print(f"data_loader {time.time() - start:.2f} Sekunden")

    # Embeddings
    print("⏳ Embeddings werden berechnet...")
    start = time.time()
    model = EmbeddingModel(config['embedding']['model'])
    embeddings = [model.get_embedding(code) for code in code_snippets]
    print(f"embedding_model {time.time() - start:.2f} Sekunden")

    # Dimensionality reduction
    start = time.time()
    method = config['dim_reduction']['method']
    n_components = min(config['dim_reduction']['params']['n_components'], len(embeddings) - 1)  # automated k determination for different solution set sizes
    random_state = config['dim_reduction']['params']['random_state']
    if method == "umap":
        n_neighbors = min(config['dim_reduction']['params']['n_neighbors'], len(embeddings) - 1)    # automated n_neighbors determination, if k is smaller than n
        reducer = DimensionalityReducer(method, params={"n_components": n_components, "n_neighbors" : n_neighbors, "random_state": random_state})
    elif method == "pca":
        reducer = DimensionalityReducer(method, params={"n_components": n_components, "random_state": random_state})    # pca doesn´t use neighbors
    reduced_embeddings = reducer.reduce(embeddings)
    print(f"Reducer {time.time() - start:.2f} Sekunden")

    # Clustering on the reduced Embeddings
    start = time.time()
    clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
    labels = clusterer.cluster(reduced_embeddings)  # colorbar values of diagram and in general the clustering allocations
    print(f"Clustering {time.time() - start:.2f} Sekunden")

    # interactive plotting (show file name by hovering)
    start = time.time()
    int_plot = InteractivePlot()
    int_plot.interactive_plot(reduced_embeddings, labels, loader.get_filenames(), loader.get_parent_dirs())
    print(f"interactive_plot {time.time() - start:.2f} Sekunden")

    # Visualization
    start = time.time()
    plotter = ClusterPlotter()
    plotter.plot(reduced_embeddings, labels)
    print(f"Visualization {time.time() - start:.2f} Sekunden")
    
    # Evaluation
    start = time.time()
    results = EvaluationMetrics.evaluate(embeddings, labels)
    print("Evaluationsergebnisse:", results)
    print("silhouette ab 0.5 und höher gut")
    print("calinski_harabasz je höher desto besser, Vgl. mit anderen Algorithmen nötig")
    print("davies_bouldin zwischen 0.3 und 0.7 gut")
    print(f"Evaluation {time.time() - start:.2f} Sekunden")

if __name__ == "__main__":
    run_pipeline()
