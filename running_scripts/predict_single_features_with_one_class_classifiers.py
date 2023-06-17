import sys

import PIL
import pandas as pd


sys.path.insert(0, '..')
from utils.apcer_metric import APCERMetric
from utils.acer_metric import ACERMetric
from utils.false_positive_rating_metric import FPRMetric
from utils.npcer_metric import NPCERMetric
from utils.true_positive_rating_metric import TPRMetric
from utils.false_rejection_rate_metric import FRRMetric

from model_architectures.one_class_classifier import OneClassClassifier
from model_architectures.clustering_method import ClusteringMethod
import yaml
import glob
import numpy as np
import tqdm
from visualization.visualize_features import Visualizer
from joblib import load


if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    print("Reading csv files containing bonafide and attack features")
    feature_files = glob.glob(f'../outputs/{params["application_parameters"]["dataset"]}/*test.csv')

    print("Checking if the prediction options are contained into the input modalities list")
    prediction_options = params["prediction_parameters"]["features_to_predict"]
    input_modalities = params["input_parameters"]["input_modalities"]

    print("Loading the PCA model used to reduce the data dimensionality")
    compressor = load("trained_models/recod-mpad_old/pca_model.joblib")
    #compressor = load("trained_models/tsne_model.joblib")

    print("Loading the clustering model")
    clustering_method = ClusteringMethod(method_name=params["clustering_parameters"]["clustering_method"],
                                         number_of_clusters=params["clustering_parameters"]["number_of_clusters"],
                                         model_path=f'trained_models/{params["application_parameters"]["dataset"]}/clustering_model_{params["clustering_parameters"]["clustering_method"]}.joblib')

    all_features = None
    label_list = []
    color_list = []
    for file in tqdm.tqdm(feature_files):
        print(f"Reading for file {file}")
        feature_file = np.genfromtxt(file, delimiter=',', dtype=np.float16, skip_header=0)

        if 'bonafide' in file:
            label_list.extend([1] * feature_file.shape[0])
            color_list.extend(['blue'] * feature_file.shape[0])
        elif 'attack_cce' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["red"] * feature_file.shape[0])
        elif 'attack_disp' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["purple"] * feature_file.shape[0])
        elif 'attack_hp' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["green"] * feature_file.shape[0])
        elif 'attack_print' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["yellow"] * feature_file.shape[0])

        if all_features is None:
            all_features = feature_file
        else:
            all_features = np.concatenate((all_features, feature_file), axis=0)
            all_features = np.nan_to_num(all_features)

    visualizer = Visualizer(input_data=all_features,
                            compress_method=params["visualization_parameters"]["compress_method"],
                            number_of_dimensions=params["visualization_parameters"]["number_of_dimensions"],
                            label_list=label_list,
                            compress_model=compressor,
                            color_list=color_list)

    print("Computing the cluster for each entry")
    compressed_features = compressor.fit_transform(all_features)
    clusters = np.array(clustering_method.model_predict(compressed_features), dtype=int)

    predictions = []
    label_total_list = []

    for i in range(0, params["clustering_parameters"]["number_of_clusters"]):
        print('Loading one-class classifiers')
        one_class_classifier = OneClassClassifier(classifier_name=params["one_class_parameters"]["classifier_name"],
                                                  classifier_path=f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{params["one_class_parameters"]["classifier_name"]}_{i}.joblib')

        try:
            indexes = np.where(clusters == i)
            predictions.extend(list(one_class_classifier.model_predict(predict_data=all_features[indexes])))
            label_total_list.extend(list(np.array(label_list, dtype=int)[indexes]))
        except:
            print(f"No data found for cluster {i}")

    correct = 0
    correct_1s = 0
    correct_0s = 0
    for i in tqdm.tqdm(range(0, len(predictions))):
        if predictions[i] == label_total_list[i]:
            correct += 1

            if label_total_list[i] == 1:
                correct_1s += 1
            else:
                correct_0s += 1

    print(f"The data contains {len(predictions)} samples")
    print(
        f"The final accuracy is {correct / len(predictions)}. The number of correct 1s is {correct_1s}, while correct 0s is {correct_0s}")

    apcer_metric = APCERMetric(metric_name="Attack_presentation_classification_error_rate",
                               input_data=np.array(predictions), label_data=np.array(label_total_list))
    npcer_metric = NPCERMetric(metric_name="Attack_presentation_classification_error_rate",
                               input_data=np.array(predictions), label_data=np.array(label_total_list))
    acer_metric = ACERMetric(metric_name="Attack_presentation_classification_error_rate",
                             input_data=np.array(predictions), label_data=np.array(label_total_list))
    fpr_metric = FPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_total_list))
    tpr_metric = TPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_total_list))
    frr_metric = FRRMetric(metric_name="False_rejection_rate",
                           input_data=np.array(predictions), label_data=np.array(label_total_list))


    print(f"APCER: {apcer_metric.metric_value}")
    print(f"NPCER: {npcer_metric.metric_value}")
    print(f"ACER: {acer_metric.metric_value}")
    print(f"FPR (FAR): {fpr_metric.metric_value}")
    print(f"FRR: {frr_metric.metric_value}")
    print(f"TPR: {tpr_metric.metric_value}")
