from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score # evaluation metrics of clustering

class EvaluationMetrics:
    @staticmethod   # static method since no object of this class is needed, because there is no need to save internal states
    def evaluate(embeddings, labels):
        results = {}
        if len(set(labels)) > 1:    # for useful evaluation there is more than one cluster needed
            results['silhouette'] = silhouette_score(embeddings, labels)
            results['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
            results['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
        else:
            results['silhouette'] = None
            results['calinski_harabasz'] = None
            results['davies_bouldin'] = None
        return results

# silhouette: Measures how similar an object is to its own cluster compared to other clusters (values from -1 to 1, higher is better)
# calinski_harabasz: Ratio of between- to within-cluster variance (the higher, the better)
# davies_bouldin: Average similarity of a cluster to its most similar cluster (the lower, the better).

# good evaluation scores:
 # silhouette: 0.5 or higher
 # calinski_harabasz: highest compared to the others
 # davies_bouldin: between 0.3 and 0.7