import numpy as np
from joblib import load
from sklearn.cluster import MiniBatchKMeans, KMeans

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
            return self.model.predict(predict_data)
        else:
            return None

    @classmethod
    def _instantiate_method(cls, method_name: str, number_of_clusters: int):
        """
        Class method that instantiates a model to perform the clustering
        """
        if method_name == 'k-means':
            #model = MiniBatchKMeans(n_clusters=number_of_clusters, random_state=0, batch_size=1024, n_init="auto")
            model = KMeans(n_clusters=number_of_clusters, random_state=0, n_init="auto")
        else:
            model = None

        return model
