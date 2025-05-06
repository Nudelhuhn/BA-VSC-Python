from sklearn.cluster import KMeans
import hdbscan

class ClusteringEngine:
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def cluster(self, embeddings):
        if self.method == "kmeans":
            model = KMeans(**self.params)
        elif self.method == "hdbscan":
            model = hdbscan.HDBSCAN(**self.params)
        else:
            raise ValueError("Unsupported clustering method.")

        labels = model.fit_predict(embeddings)
        return labels
