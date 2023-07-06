import numpy as np
from joblib import load
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN, SpectralClustering

class ClusteringMethod:

    def __init__(self, method_name: str, number_of_clusters: int, model_path: str = None):
        """
        This class represents the one-class classifiers used to classify the data.
        """
        self.method_name = method_name
        if model_path is None:
            self.model = ClusteringMethod._instantiate_method(method_name=method_name, number_of_clusters=number_of_clusters)
        else:
            self.model = load(model_path)

    def fit_model(self, input_data: np.ndarray):
        """
        Method to train the model
        """
        if self.model is not None:
            #self.model.partial_fit(input_data)
            self.model.fit(input_data)

    def retrieve_centers(self):
        """
        Method to retrieve the center of the trained clusters
        """
        if self.model is not None:
            return self.model.cluster_centers_
        else:
            return None

    def model_predict(self, predict_data: np.ndarray):
        """
        Method to perform the predictions
        """
        if self.model is not None:
            if self.method_name == 'k-means':
                return self.model.predict(predict_data)
            else:
                return self.model.labels_.astype(np.int16)
        else:
            return None

    def dbscan_predict(self, X):
        nr_samples = X.shape[0]
        y_new = np.ones(shape=nr_samples, dtype=int) * -1

        for i in range(nr_samples):
            diff = self.model.components_ - X[i, :]  # NumPy broadcasting
            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.model.eps:
                y_new[i] = self.model.labels_[self.model.core_sample_indices_[shortest_dist_idx]]

        return y_new

    @classmethod
    def _instantiate_method(cls, method_name: str, number_of_clusters: int):
        """
        Class method that instantiates a model to perform the clustering
        """
        if method_name == 'k-means':
            #model = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0, batch_size=1024, n_init="auto")
            model = KMeans(n_clusters=number_of_clusters, random_state=0, n_init="auto")
        elif method_name == 'dbscan':
            model = DBSCAN(eps=3, min_samples=2)
        elif method_name == 'spectral_clustering':
            model = SpectralClustering(n_clusters=number_of_clusters, random_state=42, assign_labels='discretize',
                                       eigen_solver='arpack', affinity='nearest_neighbors')
        else:
            model = None

        return model
