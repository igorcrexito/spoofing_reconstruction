import sys
sys.path.insert(0, '..')
import pandas as pd
import yaml
from model_architectures.one_class_classifier import OneClassClassifier
import numpy as np
from joblib import dump


def train_one_class_classifier(input_dataframe: pd.DataFrame, classifier_name: str):

    ## instantiating a one-class classifier
    one_class_classifier = OneClassClassifier(classifier_name=classifier_name)

    ## fitting the one-class model
    features = np.array(input_dataframe['features'].to_list())

    ## training the one-class classifier
    one_class_classifier.fit_model(features)

    ## saving trained models to the disk
    dump(one_class_classifier.classifier, f'trained_models/one_class_{classifier_name}.joblib')


if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ## reading the features extracted with autoencoder activations
    autoencoder_features = pd.read_json('../outputs/autoencoder_features_bonafide.json')

    ## invoking the method to train the one-class classifiers
    train_one_class_classifier(input_dataframe=autoencoder_features,
                               classifier_name=params["one_class_parameters"]["classifier_name"])
