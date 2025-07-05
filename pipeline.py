import time
complete_time = time.time()
start = time.time()
import os
import numpy as np
import yaml
import warnings
from utils.data_loader import DataLoader
from utils.config_suggester import suggest_config_settings
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
from clustering.clustering_engine import ClusteringEngine
from utils.score_binning import bin_scores
from visualization.advanced_interactive_plot import AdvancedInteractivePlot
# from visualization.cluster_plotter import ClusterPlotter
from evaluation.evaluation_metrics import EvaluationMetrics
from reporting.report_generator import ReportGenerator
print(f"imports {time.time() - start:.2f} Seconds")


start = time.time()
def load_config(path="config.yaml"):
    with open(path, "r") as file:               # open and read (r) file
        return yaml.safe_load(file)             # load file
print(f"config {time.time() - start:.2f} Seconds")


def run_pipeline():
    config = load_config()


    # ignore some unimportant warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # ignores the warning "FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8. warnings.warn("
    warnings.filterwarnings("ignore", category=UserWarning)     # ignores the warning "UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism. warn("


    # dynamically determine current path
    start = time.time()
    base_path = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath(__file__): absoluate path of __file__ (variable with current path), os.path.dirname(): path to parent folder
    input_path = os.path.join(base_path, config['data']['input_path'])
    print(f"path determination {time.time() - start} Seconds")


    # load data
    start = time.time()
    loader = DataLoader(config['data']['files_to_look_for'], input_path, config['data']['exclude_files'], config['data']['exclude_folders'])
    code_snippets = loader.load_code_files(concat=True)
    print(f"data_loader {time.time() - start:.2f} Seconds")


    # # dynamic config
    # start = time.time()
    # num_solutions = len(code_snippets)
    # suggested_config = suggest_config_settings(num_solutions)
    # config['dim_reduction'] = suggested_config['dim_reduction']
    # config['clustering'] = suggested_config['clustering']
    # config['data']['score_bins'] = suggested_config['score_bins']
    # print(f"dynamic config {time.time() - start:.2f} Seconds")


    # score bins
    start = time.time()
    scores = loader.get_scores()
    binned_scores = bin_scores(scores, config['data']['score_bins'])
    unique_bins = sorted(set(binned_scores))
    print(f"score bins {time.time() - start:.2f} Seconds")


    # Embedding
    model = EmbeddingModel(config['embedding']['model'])
    cache_path = "cached_embeddings.npy"
    
    # save all information resulting in the loop to display it all in one plot after the loop
    all_embeddings, all_labels, all_filenames, all_parent_dirs, all_bins = [], [], [], [], []

    for score_bin in unique_bins:
        print(f"\n=== Processing bin: {score_bin} ===")

        indices = [i for i, b in enumerate(binned_scores) if b == score_bin]

        if len(indices) < 4:
            print(f"Not enough solutions in bin {score_bin}, skipping...")
            continue

        snippets_bin = [code_snippets[i] for i in indices]
        filenames_bin = [loader.get_filenames()[i] for i in indices]
        parent_dirs_bin = [loader.get_parent_dirs()[i] for i in indices]

        start = time.time()
        if os.path.exists(cache_path):
            embeddings_cache = np.load(cache_path)
            if len(embeddings_cache) >= len(snippets_bin):
                # Nur die benötigten Embeddings laden
                embeddings = embeddings_cache[:len(snippets_bin)]
                print(f"✅ {len(snippets_bin)} Embeddings aus Cache geladen.")
            else:
                print("⚠️ Nicht genügend gecachte Embeddings vorhanden. Berechne neu...")
                embeddings = np.array([model.get_embedding(code) for code in snippets_bin])
                np.save(cache_path, embeddings)
                print("💾 Embeddings im Cache gespeichert.")
        else:
            print("⏳ Embeddings werden berechnet...")
            embeddings = np.array([model.get_embedding(code) for code in snippets_bin])
            np.save(cache_path, embeddings)
            print("💾 Embeddings im Cache gespeichert.")
        print(f"Embedding {time.time() - start:.2f} Seconds")


        # Dimensionality reduction
        start = time.time()
        reducer = DimensionalityReducer(config['dim_reduction']['method'], config['dim_reduction']['params'])
        reduced_embeddings = reducer.reduce(embeddings)
        print(f"Reducer {time.time() - start:.2f} Seconds")


        # Clustering on the reduced Embeddings
        start = time.time()
        clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
        labels = clusterer.cluster(reduced_embeddings)
        print(f"Clustering {time.time() - start:.2f} Seconds")

        # Evaluation
        start = time.time()
        results = EvaluationMetrics.evaluate(all_embeddings, all_labels)
        print("Evaluation results:", results)
        print(f"Evaluation {time.time() - start:.2f} Seconds")

        # save all information resulting in the loop to display it all in one plot after the loop
        all_embeddings.extend(reduced_embeddings)
        all_labels.extend(labels)
        all_filenames.extend(filenames_bin)
        all_parent_dirs.extend(parent_dirs_bin)
        all_bins.extend([score_bin] * len(labels))


    # interactive plotting (show file name by hovering)
    start = time.time()
    AdvancedInteractivePlot.adv_int_plot(all_embeddings, all_labels, all_filenames, all_parent_dirs, all_bins)
    print(f"interactive_plot {time.time() - start:.2f} Seconds")


    # # Visualization   # currently not needed if the interactive plotting is used
    # start = time.time()
    # ClusterPlotter.plot(reduced_embeddings, labels)
    # print(f"Visualization {time.time() - start:.2f} Seconds")


    # Reporting
    start = time.time()
    ReportGenerator.generate_report(all_filenames, all_parent_dirs, all_labels, all_bins, config['data']['output_path'])
    print(f"Report saved to {config['data']['output_path']}")
    print(f"Reporting {time.time() - start:.2f} Seconds")


    print(f"complete {time.time() - complete_time} Seconds")


if __name__ == "__main__":
    run_pipeline()
