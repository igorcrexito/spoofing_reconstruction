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
from utils.bpcer_metric import BPCERMetric
from joblib import load
from model_architectures.one_class_classifier import OneClassClassifier
import yaml
import glob
import numpy as np
import tqdm
from visualization.visualize_features import Visualizer
import gc

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    if params["input_parameters"]["additional_features"] != '':
        additional_feature = f'_{params["input_parameters"]["additional_features"]}'
    else:
        additional_feature = ''

    print("Reading csv files containing bonafide and attack features")
    feature_files = glob.glob(f'../outputs/{params["application_parameters"]["dataset"]}/*test{additional_feature}.csv')

    print("Loading the compressor model")
    compressor = load(f'trained_models/{params["application_parameters"]["dataset"]}/pca_model.joblib')

    all_features = None
    label_list = []
    color_list = []

    ## optional filter class to check the behavior of each class
    #filter_class = 'makeup'
    #feature_files = [x for x in feature_files if filter_class in x]

    for file in tqdm.tqdm(feature_files):
        feature_file = np.genfromtxt(file, delimiter=',', dtype=np.float16,)
        print(f'the number of entries is: {feature_file.shape[0]}')

        if 'bonafide' in file:
            label_list.extend([1] * feature_file.shape[0])
            color_list.extend(['blue'] * feature_file.shape[0])
        elif 'glasses' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["red"] * feature_file.shape[0])
        elif 'mannequin' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["purple"] * feature_file.shape[0])
        elif 'print' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["green"] * feature_file.shape[0])
        elif 'replay' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["yellow"] * feature_file.shape[0])
        elif 'rigid_mask' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["orange"] * feature_file.shape[0])
        elif 'flexible_mask' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["gray"] * feature_file.shape[0])
        elif 'paper_mask' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["black"] * feature_file.shape[0])
        elif 'wigs' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["pink"] * feature_file.shape[0])
        elif 'tattoo' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["cyan"] * feature_file.shape[0])
        elif 'makeup' in file:
            label_list.extend([-1] * feature_file.shape[0])
            color_list.extend(["magenta"] * feature_file.shape[0])

        if all_features is None:
            all_features = feature_file
        else:
            all_features = np.concatenate((all_features, feature_file), axis=0)

    visualizer = Visualizer(input_data=all_features,
                            compress_method=params["visualization_parameters"]["compress_method"],
                            number_of_dimensions=params["visualization_parameters"]["number_of_dimensions"],
                            label_list=label_list,
                            compress_model=compressor,
                            color_list=color_list)

    print('Loading one-class classifier')
    one_class_classifier = OneClassClassifier(classifier_name=params["one_class_parameters"]["classifier_name"],
                                              classifier_path=f'trained_models/{params["application_parameters"]["dataset"]}/one_class_{params["one_class_parameters"]["classifier_name"]}.joblib')

    predictions = one_class_classifier.model_predict(predict_data=all_features)
    correct = 0
    correct_1s = 0
    correct_0s = 0

    for i in tqdm.tqdm(range(0, len(predictions))):
        if predictions[i] == label_list[i]:
            correct += 1

            if label_list[i] == 1:
                correct_1s += 1
            else:
                correct_0s += 1

    print(f"The data contains {len(predictions)} samples")
    print(
        f"The final accuracy is {correct / len(predictions)}. The number of correct 1s is {correct_1s}, while correct 0s is {correct_0s}")

    apcer_metric = APCERMetric(metric_name="Attack_presentation_classification_error_rate",
                               input_data=np.array(predictions), label_data=np.array(label_list))
    frr_metric = FRRMetric(metric_name="False_rejection_rate",
                           input_data=np.array(predictions), label_data=np.array(label_list))
    fpr_metric = FPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_list))

    npcer_metric = NPCERMetric(metric_name="Attack_presentation_classification_error_rate",
                               input_data=np.array(predictions), label_data=np.array(label_list))
    tpr_metric = TPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_list))
    bpcer_metric = BPCERMetric(metric_name="Bonafide_presentation_error_metric",
                           input_data=np.array(predictions), label_data=np.array(label_list))

    print(f"APCER: {apcer_metric.metric_value}")
    print(f"NPCER: {npcer_metric.metric_value}")
    print(f"ACER: {(apcer_metric.metric_value + bpcer_metric.metric_value)/2}")
    print(f"BPCER: {bpcer_metric.metric_value}")
    print(f"FPR: {fpr_metric.metric_value}")
    print(f"FRR: {frr_metric.metric_value}")
    print(f"TPR: {tpr_metric.metric_value}")
    print(f"HTER: {(fpr_metric.metric_value + frr_metric.metric_value)/2}")
