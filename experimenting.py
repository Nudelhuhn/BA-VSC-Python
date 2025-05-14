import time
import yaml
import os
import pandas as pd
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

    # Load data
    loader = DataLoader(config['data']['file_name'], config['data']['input_path'])
    code_snippets = loader.load_code_files()
    filenames = loader.get_filenames()

    # Embedding
    model = EmbeddingModel(config['embedding']['model'])
    embeddings = np.array([model.get_embedding(code) for code in code_snippets])

    # Define possible methods and params
    dim_reduction_methods = {
        "umap": {
            "params": {
                "n_components": 2,
                "n_neighbors": 10,
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

    results = []
    cluster_assignments = []  # To store filenames + their cluster label

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

            # Track filenames and their assigned cluster for this combination
            for filename, label in zip(filenames, labels):
                cluster_assignments.append({
                    "filename": filename,
                    "dim_reduction": dr_method,
                    "clustering": cluster_method,
                    "cluster_label": label
                })

    # ensure output path exists, if not it´s created
    os.makedirs(config['data']['output_path'], exist_ok=True)

    # Save evaluation results
    output_path = os.path.join(config['data']['output_path'], "experiment_results.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)

    # Save cluster assignments
    cluster_path = os.path.join(config['data']['output_path'], "cluster_assignments.csv")
    df_clusters = pd.DataFrame(cluster_assignments)
    df_clusters.to_csv(cluster_path, index=False)

    # Drop Zeilen mit fehlenden Werten
    clean_df = df_results.dropna()

    # Normalize Scores für Ranking-Berechnung
    for metric in ['silhouette', 'calinski_harabasz']:
        clean_df[f"{metric}_norm"] = (clean_df[metric] - clean_df[metric].min()) / (clean_df[metric].max() - clean_df[metric].min())

    clean_df['davies_bouldin_norm'] = (clean_df['davies_bouldin'].max() - clean_df['davies_bouldin']) / (clean_df['davies_bouldin'].max() - clean_df['davies_bouldin'].min())

    # Gesamt-Score berechnen
    clean_df['total_score'] = clean_df[['silhouette_norm', 'calinski_harabasz_norm', 'davies_bouldin_norm']].mean(axis=1)

    # Ranking anhand des Gesamt-Scores
    ranking = clean_df.sort_values('total_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    # Nur gewünschte Spalten für das finale Ranking-CSV
    final_cols = ['dim_reduction', 'clustering', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_clusters', 'n_noise', 'rank']
    final_ranking = ranking[final_cols]

    # Exportieren
    final_ranking.to_csv(os.path.join(config['data']['output_path'], "clustering_ranking_final.csv"), index=False)

    print("\n")
    print("✔️ Clean Ranking saved.")
    print(f"Beste Kombination: {final_ranking.iloc[0]['dim_reduction']} + {final_ranking.iloc[0]['clustering']} (Rank 1)")
    print(f"📄 Evaluation results saved to: {output_path}")
    print(f"📄 Cluster assignments saved to: {cluster_path}")

if __name__ == "__main__":
    run_experiments()
