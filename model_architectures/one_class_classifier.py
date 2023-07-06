from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import numpy as np
from joblib import load


class OneClassClassifier:

    def __init__(self, classifier_name: str, classifier_path: str = None):
        """
        This class represents the one-class classifiers used to classify the data.
        """
        self.classifier_name = classifier_name
        if classifier_path is None:
            self.classifier = OneClassClassifier._instantiate_classifier(classifier_name=classifier_name)
        else:
            self.classifier = load(classifier_path)

    def fit_model(self, input_data: np.ndarray):
        """
        Method to train the model
        """
        if self.classifier is not None:
            self.classifier.fit(input_data)

    def model_predict(self, predict_data: np.ndarray):
        """
        Method to perform the predictions
        """
        if self.classifier is not None:
            return self.classifier.predict(predict_data)
        else:
            return None

    @classmethod
    def _instantiate_classifier(cls, classifier_name: str):
        """
        Class method that instantiates a model based on its name. Only SVM is supported
        """
        if classifier_name == 'svm':
            classifier = OneClassSVM(gamma=0.1, kernel='linear', nu=0.1)
        elif classifier_name == 'isolation_forest':
            classifier = IsolationForest(contamination=0.138, random_state=42)
        elif classifier_name == 'mcd':
            classifier = EllipticEnvelope(contamination=0.1, random_state=42, support_fraction=0.1)
        else:
            classifier = None

        return classifier
