import yaml
import os
import pandas as pd
import warnings
from utils.data_loader import DataLoader
from utils.score_binning import ScoreBinner
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimension_reducer import DimensionReducer
from clustering.clustering_engine import ClusteringEngine
from evaluation.evaluation_metrics import EvaluationMetrics
import numpy as np

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def run_experiments():
    config = load_config()

    # ignore warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # dynamically determine path
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, config['data']['input_path'])

    # load data
    loader = DataLoader(".java", input_path, config['data']['exclude_files'], config['data']['exclude_folders'])
    code_snippets = loader.load_code_files(concat=True)

    # score bins
    scores = loader.get_scores()
    binned_scores = ScoreBinner.bin_scores(scores, config['data']['score_bins'])
    unique_bins = sorted(set(binned_scores))

    # Embedding object
    model = EmbeddingModel(config['embedding']['model'])

    cache_path = "cached_embeddings.npy"

    results_per_combination = {}

    dim_reduction_methods = {
        "umap": {
            "params": {
                "n_components": 2,
                "n_neighbors": 15,
                "random_state": 42
            }
        },
        "pca": {
            "params": {
                "n_components": 2,
                "random_state": 42
            }
        },
        "tsne": {
            "params": {
                "n_components": 2,
                "perplexity": 30,
                "learning_rate": 200,
                "n_iter": 1000,
                "random_state": 42
            }
        }
    }

    clustering_methods = {
        "kmeans": {
            "params": {
                "n_clusters": 5,
                "random_state": 42
            }
        },
        "hdbscan": {
            "params": {
                "min_cluster_size": 2,
                "min_samples": 1,
                "cluster_selection_epsilon": 0.5,
                "metric": "euclidean"
            }
        }
    }

    # filter solutions by score bin and prepare corresponding data for clustering
    for score_bin in unique_bins:
        print(f"\n==== Processing score_bin: {score_bin} ====")
        # save indices of binned_scores which score is within the current score_bin
        indices = [i for i, b in enumerate(binned_scores) if b == score_bin]
        if len(indices) < 4:
            print(f"Not enough solutions in score_bin {score_bin}, skipping...")
            continue

        snippets_bin = [code_snippets[i] for i in indices]

        # Embedding
        if os.path.exists(cache_path):
            embeddings_cache = np.load(cache_path)
            if len(embeddings_cache) >= len(snippets_bin):
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

        # Dimension reduction
        for dr_method, dr_config in dim_reduction_methods.items():
            if dr_method == "tsne": # if the current algorithms is t-sne and ...
                n_samples = len(embeddings)
                max_perplexity = n_samples - 1 if n_samples > 1 else 1
                original_perplexity = dr_config['params']['perplexity']

                if original_perplexity >= max_perplexity:   # ... if its perplexity is too high, lower it
                    print(f"-- Perplexity {original_perplexity} too high for {n_samples} aamples - changed to {max_perplexity // 2} --")
                    dr_config['params']['perplexity'] = max(1, max_perplexity // 2)
                    
            reducer = DimensionReducer(dr_method, params=dr_config['params'])
            reduced_embeddings = reducer.reduce(embeddings)

            # Clustering of the reduced Embeddings
            for cluster_method, cluster_config in clustering_methods.items():
                clusterer = ClusteringEngine(cluster_method, cluster_config['params'])
                labels = clusterer.cluster(reduced_embeddings)
                
                # make algorithm combinations
                key = f"{dr_method}_{cluster_method}"
                if key not in results_per_combination:
                    results_per_combination[key] = {
                        "embeddings": [],
                        "labels": []
                    }

                # attach to results
                results_per_combination[key]["embeddings"].extend(reduced_embeddings)
                results_per_combination[key]["labels"].extend(labels)

                print(f"++ {dr_method} + {cluster_method} for bin {score_bin} completed ++")

    # final evaluation per combination over all collected bins
    final_results = []

    # Evaluation
    for key, data in results_per_combination.items():
        all_embeddings = np.array(data["embeddings"])
        all_labels = np.array(data["labels"])
        
        metrics = EvaluationMetrics.evaluate(all_embeddings, all_labels)

        result = {
            "combination": key,
            "silhouette": metrics['silhouette'],
            "calinski_harabasz": metrics['calinski_harabasz'],
            "davies_bouldin": metrics['davies_bouldin']
        }
        final_results.append(result)

        print(f"Evaluation finished for {key}")

    # resulting dataFrame
    df_results = pd.DataFrame(final_results)

    # normalize scores for ranking
    for metric in ['silhouette', 'calinski_harabasz']:
        df_results[f"{metric}_norm"] = (df_results[metric] - df_results[metric].min()) / (df_results[metric].max() - df_results[metric].min())
    df_results['davies_bouldin_norm'] = (df_results['davies_bouldin'].max() - df_results['davies_bouldin']) / (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min())

    # calculate total score
    df_results['total_score'] = df_results[['silhouette_norm', 'calinski_harabasz_norm', 'davies_bouldin_norm']].mean(axis=1)

    # sort ranking
    ranking = df_results.sort_values('total_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    # define columns
    final_cols = ['combination', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'rank']
    final_ranking = ranking[final_cols]

    # save ranking
    os.makedirs(config['data']['output_path'], exist_ok=True)
    final_ranking.to_csv(os.path.join(config['data']['output_path'], "clustering_ranking.csv"), index=False)

    print(f"\nFinal ranking saved at {config['data']['output_path']}")
    print(f"Best combination: {final_ranking.iloc[0]['combination']} (Rang 1)")

if __name__ == "__main__":
    run_experiments()
