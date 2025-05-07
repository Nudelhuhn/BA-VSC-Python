import matplotlib.pyplot as plt

class ClusterPlotter:
    def __init__(self):
        # Kein UMAP mehr notwendig
        pass

    def plot(self, reduced_embeddings, labels):
        # Einfach die reduzierten Embeddings plotten
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='tab10')
        plt.title("Cluster Visualisierung")
        plt.colorbar()
        plt.show()
