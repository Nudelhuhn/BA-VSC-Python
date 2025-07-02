import yaml
import os
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from clustering.clustering_engine import ClusteringEngine
from evaluation.evaluation_metrics import EvaluationMetrics
from visualization.cluster_plotter import ClusterPlotter

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()

    base_path = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath(__file__): absoluate path of __file__ (variable with current path), os.path.dirname(): path to parent folder
    input_path = os.path.join(base_path, config['data']['input_path'])

    # Daten laden
    loader = DataLoader(config['data']['files_to_look_for'], input_path, config['data']['exclude_files'], config['data']['exclude_folders'])
    code_snippets = loader.load_code_files()

    # Embeddings erzeugen
    model = EmbeddingModel(config['embedding_model'])
    embeddings = [model.get_embedding(code) for code in code_snippets]

    # Clustering
    clusterer = ClusteringEngine(config['clustering_method'], config['clustering_params'])
    labels = clusterer.cluster(embeddings)

    # Evaluation
    results = EvaluationMetrics.evaluate(embeddings, labels)
    print("Evaluationsergebnisse:", results)

    # Visualisierung
    plotter = ClusterPlotter(
        n_neighbors=config['visualization']['n_neighbors'],
        min_dist=config['visualization']['min_dist']
    )
    plotter.plot(embeddings, labels)

if __name__ == "__main__":
    run_pipeline()