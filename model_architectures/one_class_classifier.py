from sklearn.svm import OneClassSVM
import numpy as np


class OneClassClassifier:

    def __init__(self, classifier_name: str):
        """
        This class represents the one-class classifiers used to classify the data.
        Only SVM is supported for a while
        """
        self.classifier_name = classifier_name
        self.classifier = OneClassClassifier.instantiate_classifier(classifier_name=classifier_name)

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
            self.classifier.predict(predict_data)

    @classmethod
    def instantiate_classifier(cls, classifier_name: str):
        """
        Class method that instantiates a model based on its name. Only SVM is supported
        """
        if classifier_name == 'svm':
            classifier = OneClassSVM(gamma='auto')
        else:
            classifier = None

        return classifier
