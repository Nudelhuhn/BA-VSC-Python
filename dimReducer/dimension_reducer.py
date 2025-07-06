class DimensionReducer:
    def __init__(self, algorithm="umap", params=None):
        self.algorithm = algorithm
        self.params = params if params is not None else {}  # if no params given, use empty dictionary

    def reduce(self, embeddings):
        if self.algorithm == "pca":
            from sklearn.decomposition import PCA  # only import if necessary (lazy loading)
            model = PCA(**self.params)   # create object with given params
        elif self.algorithm == "tsne":
            from sklearn.manifold import TSNE
            model = TSNE(**self.params)
        elif self.algorithm == "umap":
            import umap
            model = umap.UMAP(**self.params)
        else:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: {self.algorithm}")

        reduced_embeddings = model.fit_transform(embeddings) # fit model to data, then transform it; combination of fit() and transform()
        return reduced_embeddings
