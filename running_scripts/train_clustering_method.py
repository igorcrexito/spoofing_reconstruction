import sys

sys.path.insert(0, '..')
import pandas as pd
import yaml
from model_architectures.clustering_method import ClusteringMethod
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ripser import ripser

from joblib import dump


if __name__ == '__main__':
    print("Reading the configuration yaml that stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    ## reading the features extracted with autoencoder activations
    print("Reading training file (bonafide data)")
    autoencoder_features = np.array(pd.read_csv(
        f'../outputs/{params["application_parameters"]["dataset"]}/autoencoder_features_single_model_bonafide_train.csv', header=None), dtype=np.float16)

    np.random.shuffle(autoencoder_features)

    ## invoking the method to train the one-class classifiers
    print('Training clustering method')
    batch_size = params["clustering_parameters"]["batch_size"]
    number_of_iterations = int(np.shape(autoencoder_features)[0]/batch_size)

    print("Instantiating the model")
    clustering_method = ClusteringMethod(method_name=params["clustering_parameters"]["clustering_method"],
                                         number_of_clusters=params["clustering_parameters"]["number_of_clusters"])

    print("Training the model - Keep in mind that clustering methods do not perform well in very high dimensionality data")

    compression_method = params['visualization_parameters']['compress_method']
    if compression_method == 'pca':
        compressor = PCA(n_components=params["visualization_parameters"]["number_of_dimensions"])
    elif compression_method == 'tsne':
        compressor = TSNE(n_components=params["visualization_parameters"]["number_of_dimensions"], random_state=42)

    reduced_autoencoder_features = compressor.fit_transform(autoencoder_features)

    clustering_method.fit_model(reduced_autoencoder_features)

    print("Saving PCA model to be loaded later")
    dump(compressor, f'trained_models/{params["application_parameters"]["dataset"]}/{compression_method}_model.joblib')

    print("Saving the model in the disk")
    dump(clustering_method.model, f'trained_models/{params["application_parameters"]["dataset"]}/clustering_model_{params["clustering_parameters"]["clustering_method"]}.joblib')
    print(f'Clustering method {clustering_method} is trained')