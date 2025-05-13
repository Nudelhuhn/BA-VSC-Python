import matplotlib.pyplot as plt
import hdbscan

class ClusterVisualizer:
    def test_hdbscan_cluster_counts(self, embeddings, min_cluster_sizes):
        cluster_counts = []

        for size in min_cluster_sizes:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=size, 
                                        min_samples=1, 
                                        cluster_selection_epsilon=0.5,
                                        metric="euclidean")
            labels = clusterer.fit_predict(embeddings)
            
            # Anzahl der Cluster (ohne Noise-Label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            cluster_counts.append(n_clusters)

            print(f"min_cluster_size: {size} → Cluster gefunden: {n_clusters}")

        # Plotten
        plt.figure(figsize=(8, 5))
        plt.plot(min_cluster_sizes, cluster_counts, marker='o')
        plt.title("Anzahl der Cluster in Abhängigkeit von min_cluster_size")
        plt.xlabel("min_cluster_size")
        plt.ylabel("Anzahl der erkannten Cluster")
        plt.grid(True)
        plt.show()
