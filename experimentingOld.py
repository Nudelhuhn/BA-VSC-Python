import yaml
import os
import pandas as pd
import warnings
from utils.data_loader import DataLoader
from embeddings.embedding_model import EmbeddingModel
from dimReducer.dimensionality_reducer import DimensionalityReducer
from clustering.clustering_engine import ClusteringEngine
from evaluation.evaluation_metrics import EvaluationMetrics
import numpy as np

def load_config(path="config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def run_experiments():
    config = load_config()

    # ignore some unimportant warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load data
    loader = DataLoader(".java", config['data']['input_path'], config['data']['exclude_files'])
    code_snippets = loader.load_code_files()

    # Embedding
    model = EmbeddingModel(config['embedding']['model'])
    cache_path = "cached_embeddings.npy"

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

    # Define possible methods and params
    dim_reduction_methods = {
        "umap": {
            "params": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42,
                "spread": 1.0,
                "learning_rate": 1.0,
                "init": "spectral"
            }
        },
        "pca": {
            "params": {
                "n_components": 2,
                "svd_solver": "auto",
                "random_state": 42
            }
        },
        "tsne": {
            "params": {
                "n_components": 2,
                "perplexity": 5.0, # needs to be smaller than sample size
                "learning_rate": 200.0,
                "n_iter": 1000,
                "metric": "euclidean",
                "random_state": 42,
                "init": "random",
                "early_exaggeration": 12.0,
                "angle": 0.5,
                "verbose": 0
            }
        }
    }

    clustering_methods = {
        "hdbscan": {
            "params": {
                "min_cluster_size": 2,
                "min_samples": 1,
                "cluster_selection_epsilon": 0.5,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
                "alpha": 1.0,
                "allow_single_cluster": False
            }
        },
        "kmeans": {
            "params": {
                "n_clusters": 5,
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300,
                "random_state": 42
            }
        }
    }

    results = []

    for dr_method, dr_config in dim_reduction_methods.items():
        # Dimensionality Reduction
        reducer = DimensionalityReducer(dr_method, params=dr_config['params'])
        reduced_embeddings = reducer.reduce(embeddings)

        for cluster_method, cluster_config in clustering_methods.items():
            # Clustering
            clusterer = ClusteringEngine(cluster_method, cluster_config['params'])
            labels = clusterer.cluster(reduced_embeddings)

            # Evaluation
            metrics = EvaluationMetrics.evaluate(reduced_embeddings, labels)

            # Cluster count
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            # Collect result
            result = {
                "dim_reduction": dr_method,
                "clustering": cluster_method,
                "silhouette": metrics['silhouette'],
                "calinski_harabasz": metrics['calinski_harabasz'],
                "davies_bouldin": metrics['davies_bouldin'],
                "n_clusters": n_clusters,
                "n_noise": n_noise
            }
            results.append(result)
            print(f"✔️ {dr_method} + {cluster_method} done.")


    # ensure output path exists, if not it´s created
    os.makedirs(config['data']['output_path'], exist_ok=True)

    # Save evaluation results
    df_results = pd.DataFrame(results)

    # Normalize Scores für Ranking-Berechnung
    for metric in ['silhouette', 'calinski_harabasz']:
            df_results[f"{metric}_norm"] = (df_results[metric] - df_results[metric].min()) / (df_results[metric].max() - df_results[metric].min())
    
    df_results['davies_bouldin_norm'] = (df_results['davies_bouldin'].max() - df_results['davies_bouldin']) / (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min())
    
    # Gesamt-Score berechnen
    df_results['total_score'] = df_results[['silhouette_norm', 'calinski_harabasz_norm', 'davies_bouldin_norm']].mean(axis=1)

    # Ranking anhand des Gesamt-Scores
    ranking = df_results.sort_values('total_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    # Nur gewünschte Spalten für das finale Ranking-CSV
    final_cols = ['dim_reduction', 'clustering', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_clusters', 'n_noise', 'rank']
    final_ranking = ranking[final_cols]

    # Exportieren
    final_ranking.to_csv(os.path.join(config['data']['output_path'], "clustering_ranking_final.csv"), index=False)

    print(f"\n✔️ Final Ranking saved to {config['data']['output_path']}")
    print(f"Beste Kombination: {final_ranking.iloc[0]['dim_reduction']} + {final_ranking.iloc[0]['clustering']} (Rank 1)")

if __name__ == "__main__":
    run_experiments()