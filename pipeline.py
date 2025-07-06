import time

complete_time = time.time()
importTime = time.time()

import os
import yaml
import warnings
import numpy as np
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimension_reducer import DimensionReducer
from clustering.clustering_engine import ClusteringEngine
from utils.score_binning import ScoreBinner
from visualization.advanced_interactive_plot import AdvancedInteractivePlot
# from visualization.cluster_plotter import ClusterPlotter
from evaluation.evaluation_metrics import EvaluationMetrics
from reporting.report_generator import ReportGenerator
print(f"imports {time.time() - importTime:.2f} Seconds")


# load config
def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def run_pipeline():
    config = load_config()


    # ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # ignores the warning "FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8. warnings.warn("
    warnings.filterwarnings("ignore", category=UserWarning)     # ignores the warning "UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism. warn("


    # dynamically determine path
    base_path = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath(__file__): absoluate path of __file__ (variable with current path), os.path.dirname(): path to parent folder
    input_path = os.path.join(base_path, config['data']['input_path'])


    # load data
    load_data_time = time.time()
    loader = DataLoader(config['data']['files_to_look_for'], input_path, config['data']['exclude_files'], config['data']['exclude_folders'])
    code_snippets = loader.load_code_files(concat=True)
    print(f"data_loader {time.time() - load_data_time:.2f} Seconds")


    # score bins
    score_bins_time = time.time()
    scores = loader.get_scores()
    binned_scores = ScoreBinner.bin_scores(scores, config['data']['score_bins'])
    unique_bins = sorted(set(binned_scores))
    print(f"score bins {time.time() - score_bins_time:.2f} Seconds")


    # Embedding object
    model = EmbeddingModel(config['embedding']['model'])
    cache_path = "cached_embeddings.npy"
    
    # save all information resulting in the loop to display it all in one plot after the loop
    all_embeddings, all_labels, all_filenames, all_parent_dirs, all_score_bins = [], [], [], [], []

    # filter solutions by score bin and prepare corresponding data for clustering
    for score_bin in unique_bins:
        print(f"\n==== Processing bin: {score_bin} ====")

        indices = [i for i, b in enumerate(binned_scores) if b == score_bin]

        if len(indices) < 4:
            print(f"Not enough solutions in bin {score_bin}, skipping...")
            continue

        snippets_bin = [code_snippets[i] for i in indices]
        filenames_bin = [loader.get_filenames()[i] for i in indices]
        parent_dirs_bin = [loader.get_parent_dirs()[i] for i in indices]


        # Embedding in loop
        embedding_time = time.time()
        if os.path.exists(cache_path):
            embeddings_cache = np.load(cache_path)
            if len(embeddings_cache) >= len(snippets_bin):
                # Nur die benötigten Embeddings laden
                embeddings = embeddings_cache[:len(snippets_bin)]
                print(f"++ {len(snippets_bin)} embeddings loaded out of cache ++")
            else:
                print("** not enough embeddings in cache, recalculate embeddings... **")
                embeddings = np.array([model.get_embedding(code) for code in snippets_bin])
                np.save(cache_path, embeddings)
                print("++ embeddings saved in cache ++")
        else:
            print("** calculate embeddings... **")
            embeddings = np.array([model.get_embedding(code) for code in snippets_bin])
            np.save(cache_path, embeddings)
            print("++ embeddings saved in cache ++")
        print(f"Embedding {time.time() - embedding_time:.2f} Seconds")


        # Dimension reduction
        dimension_reduction_time = time.time()
        reducer = DimensionReducer(config['dim_reduction']['method'], config['dim_reduction']['params'])
        reduced_embeddings = reducer.reduce(embeddings)
        print(f"Reducer {time.time() - dimension_reduction_time:.2f} Seconds")


        # Clustering of the reduced Embeddings
        clustering_time = time.time()
        clusterer = ClusteringEngine(config['clustering']['method'], config['clustering']['params'])
        labels = clusterer.cluster(reduced_embeddings)
        print(f"Clustering {time.time() - clustering_time:.2f} Seconds")

        # Evaluation
        evaluation_time = time.time()
        results = EvaluationMetrics.evaluate(reduced_embeddings, labels)
        print("Evaluation results:", results)
        print(f"Evaluation {time.time() - evaluation_time:.2f} Seconds")

        # save all information resulting in the loop to display it all in one plot after the loop
        all_embeddings.extend(reduced_embeddings)
        all_labels.extend(labels)
        all_filenames.extend(filenames_bin)
        all_parent_dirs.extend(parent_dirs_bin)
        all_score_bins.extend([score_bin] * len(labels))


    # Advanced interactive plot
    advanced_iteractive_plot_time = time.time()
    AdvancedInteractivePlot.plot(all_embeddings, all_labels, all_filenames, all_parent_dirs, all_score_bins)
    print(f"interactive_plot {time.time() - advanced_iteractive_plot_time:.2f} Seconds")


    # # Simple plot
    # cluster_plotter_time = time.time()
    # ClusterPlotter.plot(reduced_embeddings, all_labels)
    # print(f"Visualization {time.time() - cluster_plotter_time:.2f} Seconds")


    # Reporting
    reporting_time = time.time()
    ReportGenerator.generate_report(all_filenames, all_parent_dirs, all_labels, all_score_bins, config['data']['output_path'])
    print(f"Report saved to {config['data']['output_path']}")
    print(f"Reporting {time.time() - reporting_time:.2f} Seconds")


    print(f"complete {time.time() - complete_time} Seconds")


if __name__ == "__main__":
    run_pipeline()
