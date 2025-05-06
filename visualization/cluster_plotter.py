import umap
import matplotlib.pyplot as plt

class ClusterPlotter:
    def __init__(self, n_neighbors=15, min_dist=0.1):
        self.reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

    def plot(self, embeddings, labels):
        embedding_2d = self.reducer.fit_transform(embeddings)
        plt.figure(figsize=(10, 7))
        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10')
        plt.title("Cluster Visualisierung")
        plt.colorbar()
        plt.show()
