class ClusteringEngine:
    def __init__(self, method, params):
        self.method = method
        self.params = params

    def cluster(self, embeddings):  # method to cluster the embeddings
        if self.method == "kmeans":
            from sklearn.cluster import KMeans  # only import if necessary
            model = KMeans(**self.params)   # create object with given params; in detail e.g. KMeans(n_clusters=3, random_state=42)
        elif self.method == "hdbscan":
            import hdbscan
            model = hdbscan.HDBSCAN(**self.params)
        else:
            raise ValueError("Unsupported clustering method.")

        labels = model.fit_predict(embeddings)  # compute cluster centers and predict cluster index for each sample; combination of fit() and predict()
        return labels
