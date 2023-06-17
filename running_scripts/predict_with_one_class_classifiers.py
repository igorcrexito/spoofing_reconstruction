import sys

import PIL
import pandas as pd

sys.path.insert(0, '..')
from utils.apcer_metric import APCERMetric
from utils.acer_metric import ACERMetric
from utils.false_positive_rating_metric import FPRMetric
from utils.npcer_metric import NPCERMetric
from utils.true_positive_rating_metric import TPRMetric

from model_architectures.one_class_classifier import OneClassClassifier
import yaml
import glob
import numpy as np
import tqdm
from visualization.visualize_features import Visualizer
import gc

FEATURE_SIZE = 6272  ## size of the feature for each modality

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    print("Reading csv files containing bonafide and attack features")
    feature_files = glob.glob("../outputs/*test.csv")

    print("Checking if the prediction options are contained into the input modalities list")
    prediction_options = params["prediction_parameters"]["features_to_predict"]
    input_modalities = params["input_parameters"]["input_modalities"]

    ## retrieving the position of the feature among the feature array
    modality_dict = {}
    for index, modality in enumerate(prediction_options):
        if modality not in input_modalities:
            raise ValueError(f"The modality {modality} is not one of the input_modalities")
        else:
            modality_dict[modality] = input_modalities.index(modality)

    all_features = None
    label_list = []
    color_list = []
    for file in tqdm.tqdm(feature_files):
        feature_file = np.genfromtxt(file, delimiter=',', dtype=np.float16, skip_header=2000)

        ## performing the prediction with features of some modalities
        if len(modality_dict.keys()) != len(input_modalities):
            partial_features = None
            for prediction_option in prediction_options:
                modality_feature = feature_file[:, modality_dict[prediction_option] * FEATURE_SIZE:(modality_dict[
                                                                                                        prediction_option] + 1) * FEATURE_SIZE]
                if partial_features is None:
                    partial_features = modality_feature
                else:
                    partial_features = np.concatenate([partial_features, modality_feature], axis=1)
                    # partial_features = np.multiply(partial_features, modality_feature)

            ## cleaning memory
            feature_file = partial_features
            partial_features = None
            gc.collect()

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

    visualizer = Visualizer(input_data=all_features,
                            compress_method=params["visualization_parameters"]["compress_method"],
                            number_of_dimensions=params["visualization_parameters"]["number_of_dimensions"],
                            label_list=label_list,
                            color_list=color_list)

    print('Loading one-class classifier')
    one_class_classifier = OneClassClassifier(classifier_name=params["one_class_parameters"]["classifier_name"],
                                              classifier_path=f'trained_models/one_class_{params["one_class_parameters"]["classifier_name"]}.joblib')

    predictions = one_class_classifier.model_predict(predict_data=all_features)
    correct = 0
    correct_1s = 0
    correct_0s = 0

    __import__("IPython").embed()
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
    npcer_metric = NPCERMetric(metric_name="Attack_presentation_classification_error_rate",
                               input_data=np.array(predictions), label_data=np.array(label_list))
    acer_metric = ACERMetric(metric_name="Attack_presentation_classification_error_rate",
                             input_data=np.array(predictions), label_data=np.array(label_list))
    fpr_metric = FPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_list))
    tpr_metric = TPRMetric(metric_name="Attack_presentation_classification_error_rate",
                           input_data=np.array(predictions), label_data=np.array(label_list))

    print(f"APCER: {apcer_metric.metric_value}")
    print(f"NPCER: {npcer_metric.metric_value}")
    print(f"ACER: {acer_metric.metric_value}")
    print(f"FPR: {fpr_metric.metric_value}")
    print(f"TPR: {tpr_metric.metric_value}")
