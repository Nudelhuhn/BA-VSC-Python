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

    # dynamically determine current path
    base_path = os.path.dirname(os.path.abspath(__file__))  # os.path.abspath(__file__): absoluate path of __file__ (variable with current path), os.path.dirname(): path to parent folder
    input_path = os.path.join(base_path, config['data']['input_path'])
    output_path = os.path.join(base_path, config['data']['output_path'])

    # Load data
    loader = DataLoader(config['data']['file_names'][0], input_path, config['data']['exclude_files'])
    code_snippets = loader.load_code_files()

    # Embedding
    model = EmbeddingModel(config['embedding']['model'])
    embeddings = np.array([model.get_embedding(code) for code in code_snippets])

    # Define possible methods and params
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

            # Collect result
            result = {
                "dim_reduction": dr_method,
                "clustering": cluster_method,
                "silhouette": metrics['silhouette'],
                "calinski_harabasz": metrics['calinski_harabasz'],
                "davies_bouldin": metrics['davies_bouldin'],
            }
            results.append(result)
            print(f"✔️ {dr_method} + {cluster_method} done.")

    # ensure output path exists, if not it´s created
    os.makedirs(output_path, exist_ok=True)

    # save evaluation results
    df_results = pd.DataFrame(results)

    # normalize Scores for ranking calculation
    for metric in ['silhouette', 'calinski_harabasz']:
        df_results[f"{metric}_norm"] = (df_results[metric] - df_results[metric].min()) / (df_results[metric].max() - df_results[metric].min())
    df_results['davies_bouldin_norm'] = (df_results['davies_bouldin'].max() - df_results['davies_bouldin']) / (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min())

    # mean value of the normalized metrics as total score
    df_results['total_score'] = df_results[['silhouette_norm', 'calinski_harabasz_norm', 'davies_bouldin_norm']].mean(axis=1)

    # sort all results by total score and assign ranks
    ranking = df_results.sort_values('total_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    # only desired columns for final ranking csv
    final_cols = ['dim_reduction', 'clustering', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'rank']
    final_ranking = ranking[final_cols]

    # export
    final_ranking.to_csv(os.path.join(output_path, "clustering_ranking.csv"), index=False)

    print(f"\nFinal Ranking saved to {output_path}")
    print(f"Best combination: {final_ranking.iloc[0]['dim_reduction']} + {final_ranking.iloc[0]['clustering']} (Rank 1)")

if __name__ == "__main__":
    run_experiments()
