import yaml
import os
import pandas as pd
import warnings
from utils.data_loader import DataLoader
from utils.score_binning import bin_scores
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

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    base_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_path, config['data']['input_path'])

    loader = DataLoader(".java", input_path, config['data']['exclude_files'], config['data']['exclude_folders'])
    code_snippets = loader.load_code_files(concat=True)

    scores = loader.get_scores()
    binned_scores = bin_scores(scores, config['data']['score_bins'])
    unique_bins = sorted(set(binned_scores))

    model = EmbeddingModel(config['embedding']['model'])

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

    # Ergebnisse pro Kombination sammeln
    results_per_combination = {}

    for score_bin in unique_bins:
        print(f"\n=== Processing bin: {score_bin} ===")

        indices = [i for i, b in enumerate(binned_scores) if b == score_bin]
        if len(indices) < 4:
            print(f"Not enough solutions in bin {score_bin}, skipping...")
            continue

        # Nur die Snippets des aktuellen Bins einbetten
        snippets_bin = [code_snippets[i] for i in indices]
        embeddings = np.array([model.get_embedding(code) for code in snippets_bin])

        for dr_method, dr_config in dim_reduction_methods.items():
            # Prüfen, ob es sich um t-SNE handelt
            if dr_method == "tsne":
                n_samples = len(embeddings)
                max_perplexity = n_samples - 1 if n_samples > 1 else 1
                original_perplexity = dr_config['params']['perplexity']

                # Wenn aktuelle perplexity zu groß, anpassen
                if original_perplexity >= max_perplexity:
                    print(f"⚠️  Perplexity {original_perplexity} zu hoch für {n_samples} Samples — setze auf {max_perplexity // 2}")
                    dr_config['params']['perplexity'] = max(1, max_perplexity // 2)
                    
            reducer = DimensionalityReducer(dr_method, params=dr_config['params'])
            reduced_embeddings = reducer.reduce(embeddings)

            for cluster_method, cluster_config in clustering_methods.items():
                clusterer = ClusteringEngine(cluster_method, cluster_config['params'])
                labels = clusterer.cluster(reduced_embeddings)

                key = f"{dr_method}_{cluster_method}"
                if key not in results_per_combination:
                    results_per_combination[key] = {
                        "embeddings": [],
                        "labels": []
                    }

                # Ergebnisse anhängen
                results_per_combination[key]["embeddings"].extend(reduced_embeddings)
                results_per_combination[key]["labels"].extend(labels)

                print(f"✔️ {dr_method} + {cluster_method} für Bin {score_bin} abgeschlossen.")

    # Endgültige Evaluation je Kombination über alle gesammelten Bins
    final_results = []

    for key, data in results_per_combination.items():
        embeddings_all = np.array(data["embeddings"])
        labels_all = np.array(data["labels"])

        # Evaluation durchführen
        metrics = EvaluationMetrics.evaluate(embeddings_all, labels_all)

        result = {
            "combination": key,
            "silhouette": metrics['silhouette'],
            "calinski_harabasz": metrics['calinski_harabasz'],
            "davies_bouldin": metrics['davies_bouldin']
        }
        final_results.append(result)

        print(f"✅ Evaluation abgeschlossen für {key}")

    # Ergebnis-DataFrame
    df_results = pd.DataFrame(final_results)

    # Scores normalisieren für Ranking
    for metric in ['silhouette', 'calinski_harabasz']:
        df_results[f"{metric}_norm"] = (df_results[metric] - df_results[metric].min()) / (df_results[metric].max() - df_results[metric].min())
    df_results['davies_bouldin_norm'] = (df_results['davies_bouldin'].max() - df_results['davies_bouldin']) / (df_results['davies_bouldin'].max() - df_results['davies_bouldin'].min())

    # Gesamtscore berechnen
    df_results['total_score'] = df_results[['silhouette_norm', 'calinski_harabasz_norm', 'davies_bouldin_norm']].mean(axis=1)

    # Ranking sortieren und speichern
    ranking = df_results.sort_values('total_score', ascending=False).reset_index(drop=True)
    ranking['rank'] = ranking.index + 1

    # Nur relevante Spalten
    final_cols = ['combination', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'rank']
    final_ranking = ranking[final_cols]

    os.makedirs(config['data']['output_path'], exist_ok=True)
    final_ranking.to_csv(os.path.join(config['data']['output_path'], "clustering_ranking.csv"), index=False)

    print(f"\n📊 Finales Ranking gespeichert unter {config['data']['output_path']}")
    print(f"🏆 Beste Kombination: {final_ranking.iloc[0]['combination']} (Rang 1)")

if __name__ == "__main__":
    run_experiments()
