class ClusteringEngine:
    def __init__(self, algorithm="hdbscan", params=None):
        self.algorithm = algorithm
        self.params = params if params is not None else {}  # if no params given, use empty dictionary

    def cluster(self, reduced_embeddings):  # algorithm to cluster the reduced_embeddings
        if self.algorithm == "kmeans":
            from sklearn.cluster import KMeans  # only import if necessary (lazy loading)
            model = KMeans(**self.params)   # create object with given params; in detail e.g. KMeans(n_clusters=3, random_state=42)
        elif self.algorithm == "hdbscan":
            import hdbscan
            model = hdbscan.HDBSCAN(**self.params)
        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

        labels = model.fit_predict(reduced_embeddings)  # compute cluster centers and predict cluster index for each sample; combination of fit() and predict()
        return labels
