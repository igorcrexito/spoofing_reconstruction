import sys

sys.path.insert(0, '..')
import pandas as pd
import yaml
from model_architectures.one_class_classifier import OneClassClassifier
from model_architectures.clustering_method import ClusteringMethod
import numpy as np
from joblib import dump, load


def train_one_class_classifier(input_dataframe: pd.DataFrame, classifier_name: str, classifier_index: int = None):
    ## instantiating a one-class classifier
    one_class_classifier = OneClassClassifier(classifier_name=classifier_name)

    ## fitting the one-class model
    features = np.array(input_dataframe)

    ## training the one-class classifier
    one_class_classifier.fit_model(features)

    ## saving trained models to the disk
    if classifier_index is not None:
        dump(one_class_classifier.classifier, f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{classifier_name}_{classifier_index}.joblib')
    else:
        dump(one_class_classifier.classifier, f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{classifier_name}.joblib')
    print(f'One class {classifier_name} is trained')


if __name__ == '__main__':
    print("Reading the configuration yaml that stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ## retrieving the specified number of clusters
    number_of_clusters = params["clustering_parameters"]["number_of_clusters"]

    print("Reading the training data files")
    ## reading the features extracted with autoencoder activations
    autoencoder_features = np.array(pd.read_csv(
        f'../outputs/{params["application_parameters"]["dataset"]}/autoencoder_features_single_model_bonafide_train.csv', header=None), dtype=np.float16)

    #class_vector = ['glasses', 'mannequin', 'print', 'replay', 'rigid_mask']
    class_vector = []
    for classe in class_vector:
        current_features = np.array(pd.read_csv(
            f'../outputs/{params["application_parameters"]["dataset"]}/autoencoder_features_single_model_{classe}_train.csv',
            header=None), dtype=np.float16)
        autoencoder_features = np.concatenate((autoencoder_features, current_features), axis=0)

    print('Reading the dimensionality reduction model and transforming features')
    ## reducing the data dimensionality and getting the clustering predictions
    compressor = load(f'trained_models/{params["application_parameters"]["dataset"]}/pca_model.joblib')
    reduced_autoencoder_features = compressor.transform(autoencoder_features)

    ## loading the clustering model
    if number_of_clusters > 1:
        print("Loading the clustering model to perform the clustering predictions")
        clustering_method = ClusteringMethod(method_name=params["clustering_parameters"]["clustering_method"],
                                             number_of_clusters=params["clustering_parameters"]["number_of_clusters"],
                                             model_path=f'trained_models/{params["application_parameters"]["dataset"]}/clustering_model_{params["clustering_parameters"]["clustering_method"]}.joblib')

        clusters = np.array(clustering_method.model_predict(reduced_autoencoder_features), dtype=int)

        ## training a one-class-classifier for each cluster
        for i in range(0, number_of_clusters):
            print(f'Training the model {params["one_class_parameters"]["classifier_name"]} of index {i}')
            indexes = np.where(clusters == i)
            train_one_class_classifier(input_dataframe=autoencoder_features[indexes],
                                       classifier_name=params["one_class_parameters"]["classifier_name"],
                                       classifier_index=i)
    else:
        train_one_class_classifier(input_dataframe=autoencoder_features,
                                   classifier_name=params["one_class_parameters"]["classifier_name"])