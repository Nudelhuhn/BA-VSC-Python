#debugging
import time
start = time.time()
#debugging
import yaml
import warnings
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
from clusterVisualizer.test_hdbscan_cluster_counts import ClusterVisualizer
from clustering.clustering_engine import ClusteringEngine
from interactivePlot.interactive_plot import InteractivePlot
from evaluation.evaluation_metrics import EvaluationMetrics
from visualization.cluster_plotter import ClusterPlotter
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

    # Daten laden
    start = time.time()
    loader = DataLoader(config['data']['file_name'], config['data']['input_path'])
    code_snippets = loader.load_code_files()
    print(f"data_loader {time.time() - start:.2f} Sekunden")

    # Embeddings erzeugen
    print("⏳ Embeddings werden berechnet...")
    start = time.time()
    model = EmbeddingModel(config['embedding']['model'])
    embeddings = [model.get_embedding(code) for code in code_snippets]
    print(f"embedding_model {time.time() - start:.2f} Sekunden")

    # Dimensionality Reduction
    # embeddings = np.random.rand(16, 768)
    start = time.time()
    reducer = DimensionalityReducer(config['dim_reduction']['method'],
                                    params={"n_components": min(config['dim_reduction']['params']['n_components'], len(embeddings) - 1),    # automated k determination for different solution set sizes
                                            "n_neighbors" : min(config['dim_reduction']['params']['n_neighbors'], len(embeddings) - 1), # automated n_neighbors determination, if k is smaller than n
                                            "random_state": config['dim_reduction']['params']['random_state']})
    # print("Shape der Embeddings vor der Dimensionsreduktion:", embeddings.shape)
    reduced_embeddings = reducer.reduce(embeddings)
    # print("Shape der Embeddings nach der Dimensionsreduktion:", reduced_embeddings.shape)
    print(f"reducer {time.time() - start:.2f} Sekunden")

    # different cluster sizes testing
    start = time.time()
    min_cluster_sizes = range(2, 15)
    visualizer = ClusterVisualizer()
    visualizer.test_hdbscan_cluster_counts(reduced_embeddings, min_cluster_sizes)
    print(f"test_hdbscan_cluster_counts {time.time() - start:.2f} Sekunden")

    # Clustering on the reduced Embeddings
    start = time.time()
    clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
    labels = clusterer.cluster(reduced_embeddings)  # colorbar values
    # unique_labels, counts = np.unique(labels, return_counts=True)
    # print("Cluster-Labels:", labels)
    # print("Anzahl Cluster:", len(set(labels)) - (1 if -1 in labels else 0))
    # print("Label-Verteilung:", dict(zip(unique_labels, counts)))
    print(f"clustering {time.time() - start:.2f} Sekunden")

    # interactive plotting (show file name by hovering)
    start = time.time()
    int_plot = InteractivePlot()
    int_plot.interactive_plot(reduced_embeddings, labels, loader.get_filenames())
    print(f"interactive_plot {time.time() - start:.2f} Sekunden")

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
