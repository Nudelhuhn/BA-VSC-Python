class DimensionalityReducer:
    def __init__(self, method="pca", params=None):
        self.method = method.lower()
        self.params = params if params is not None else {}
        self.model = None  # Modell wird erst in _initialize_model() erzeugt

    def _initialize_model(self):
        if self.method == "pca":
            from sklearn.decomposition import PCA
            return PCA(**self.params)
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
        return self.model.fit_transform(embeddings)
