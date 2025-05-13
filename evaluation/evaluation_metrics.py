from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class EvaluationMetrics:
    @staticmethod
    def evaluate(embeddings, labels):
        results = {}
        if len(set(labels)) > 1:    # are there more than one cluster? only useful if there are more than two
            results['silhouette'] = silhouette_score(embeddings, labels)
            results['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
            results['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        else:
            results['silhouette'] = None
            results['calinski_harabasz'] = None
            results['davies_bouldin'] = None
        return results
