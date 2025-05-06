import yaml
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from clustering.clustering_engine import ClusteringEngine
from evaluation.evaluation_metrics import EvaluationMetrics
from visualization.cluster_plotter import ClusterPlotter

def load_config(path="config.yaml"):
    with open(path, "r") as file:               # open and read (r) file
        return yaml.safe_load(file)             # load file

def run_pipeline():
    config = load_config()

    # Daten laden
    loader = DataLoader(config['data']['input_path'])
    code_snippets = loader.load_code_files()

    # # Embeddings erzeugen
    # model = EmbeddingModel(config['embedding_model'])
    # embeddings = [model.get_embedding(code) for code in code_snippets]

    # # Clustering
    # clusterer = ClusteringEngine(config['clustering_method'], config['clustering_params'])
    # labels = clusterer.cluster(embeddings)

    # # Evaluation
    # results = EvaluationMetrics.evaluate(embeddings, labels)
    # print("Evaluationsergebnisse:", results)

    # # Visualisierung
    # plotter = ClusterPlotter(
    #     n_neighbors=config['visualization']['n_neighbors'],
    #     min_dist=config['visualization']['min_dist']
    # )
    # plotter.plot(embeddings, labels)

if __name__ == "__main__":
    run_pipeline()
