import sys
sys.path.insert(0, '..')
import pandas as pd
import yaml
from model_architectures.one_class_classifier import OneClassClassifier
import numpy as np
from joblib import dump, load
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import tqdm


def train_one_class_classifier(input_dataframe: pd.DataFrame, classifier_name: str, classifier_index: int = None):
    ## instantiating a one-class classifier
    one_class_classifier = OneClassClassifier(classifier_name=classifier_name)

    ## fitting the one-class model
    features = np.array(input_dataframe)

    ## training the one-class classifier
    one_class_classifier.fit_model(features)

    ## saving trained models to the disk
    if classifier_index is not None:
        dump(one_class_classifier.classifier,
             f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{classifier_name}_{classifier_index}.joblib')
    else:
        dump(one_class_classifier.classifier,
             f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{classifier_name}.joblib')
    print(f'One class {classifier_name} is trained')


if __name__ == '__main__':
    print("Reading the configuration yaml that stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    additional_feature = ''

    print("Reading the training data files")
    ## reading the features extracted with autoencoder activations
    autoencoder_features = np.array(pd.read_csv(
        f'../outputs/{params["application_parameters"]["dataset"]}/autoencoder_features_single_model_bonafide_val{additional_feature}.csv',
        header=None), dtype=np.float16)

    print('Reading the dimensionality reduction model and transforming features')
    ## reducing the data dimensionality and getting the clustering predictions
    compressor = load(f'trained_models/{params["application_parameters"]["dataset"]}/pca_model.joblib')
    reduced_autoencoder_features = compressor.transform(autoencoder_features)

    ## computing silhouette score
    best_score = -1
    best_contamination = 0

    contamination_list = []
    score_list = []

    ## loading the clustering model
    for i in tqdm.tqdm(np.linspace(0.01, 0.3, 30)):
        clf = IsolationForest(contamination=np.round(i, 2))
        preds = clf.fit_predict(autoencoder_features)
        score = silhouette_score(autoencoder_features, preds)

        if score > best_score:
            best_score = score
            best_contamination = i

        score_list.append(score)
        contamination_list.append(i)

    fig, ax = plt.subplots()

    ax.bar(np.linspace(0, 30, 30), score_list)
    ax.set_xticks(np.linspace(0, 30, 30))
    ax.set_xticklabels([np.round(x,2) for x in contamination_list])
    plt.show()

    __import__("IPython").embed()