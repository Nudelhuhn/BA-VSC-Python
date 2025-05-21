class DimensionalityReducer:
    def __init__(self, method="pca", params=None):  # pca as default value and optional params
        self.method = method.lower()
        self.params = params if params is not None else {}  # if no params given, use empty dictionary
        self.model = None  # assign model in initialize method

    def _initialize_model(self):
        if self.method == "pca":
            from sklearn.decomposition import PCA   # only import if necessary
            return PCA(**self.params)   # create object with given params
        elif self.method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(**self.params)
        elif self.method == "umap":
            import umap
            return umap.UMAP(**self.params)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.method}")

    def reduce(self, embeddings):
        if self.model is None:
            self.model = self._initialize_model()
        return self.model.fit_transform(embeddings) # fit model to data, then transform it; combination of fit() and transform()
