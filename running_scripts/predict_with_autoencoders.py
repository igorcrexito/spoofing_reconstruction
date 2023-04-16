import sys

import PIL
import pandas as pd

sys.path.insert(0, '..')

from dataset.image_dataset import ImageDataset
from descriptors.bsif_descriptor import BSIFDescriptor
from model_architectures.autoencoder import Autoencoder
from PIL import Image
import numpy as np
import os
import yaml

autoencoders_base_path = 'trained_models/'


def _normalize_image(image: PIL.Image.Image):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """

    # image = np.array(image.convert('RGBA'))

    image = np.array(image).astype('float')

    for i in range(3):
        minval = image[..., i].min()
        maxval = image[..., i].max()
        if minval != maxval:
            # image[..., i] -= minval
            image[..., i] /= maxval
    # return Image.fromarray(image.astype('uint8'), 'RGBA')
    return image


def predict_image_class(image: PIL.Image.Image, autoencoder_list: list, class_dictionary: dict):
    """
    This method predicts the image class based on the reconstruction error of every model
    """
    min_error = 999999999999  ## setting a very high error level
    min_index = -1  ## random index value to be posteriorly overwritten

    for index, model in enumerate(autoencoder_list):

        ## reconstructing the image
        reconstructed_image = model.model.predict(image)
        reconstruction_error = int(abs(np.array(reconstructed_image) - np.array(image)).sum())

        if reconstruction_error < min_error:
            min_error = reconstruction_error
            min_index = index

    class_name = class_dictionary[index]

    return {'class_name': class_name, 'reconstruction_error': min_error, 'class_prediction': min_index}


def predict_images(image_base_path: str, input_modality_list: list, image_size: tuple = (224, 224), partial_activations: str = ""):
    image_list = os.listdir(image_base_path)

    ae_list = []
    print('Loading previously trained Autoencoder models')
    for modality in input_modality_list:
        autoencoder_model = Autoencoder(input_modality=modality,
                                        input_dimension=image_size[0],
                                        summarize_model=False,
                                        intermediate_layer=partial_activations)
        autoencoder_model.model.load_weights(f"trained_models/best_autoencoder_{modality}.hdf5")
        ae_list.append(autoencoder_model)

    model_responses = []
    image_full_path_list = []
    for index, image_path in enumerate(image_list):
        image_full_path = f"{image_base_path}/{image_path}"
        image = Image.open(image_full_path).resize((image_size[0], image_size[1]))
        image = _normalize_image(image=np.array(image))
        image = np.reshape(image, (1, image_size[0], image_size[1], 3))

        if partial_activations != "":
            if index == 0: print(f'This is a partial prediction, retrieving activations from: {partial_activations}')

            ae_activation = []
            for model in ae_list:
                ae_activation.append(model.partial_model.predict(image)[0])

            ae_activation = np.array(ae_activation).flatten()
            model_responses.append(ae_activation)
            image_full_path_list.append(image_full_path)
        else:
            status_string = f'This is a full prediction, retrieving the output label obtained with the models'
            if index == 0: print(status_string)
            output = predict_image_class(image=image, autoencoder_list=ae_list, class_dictionary={1: 'rgb', 2: 'noise'})
            model_responses.append(output)
            image_full_path_list.append(image_full_path)

        ## writing the output dataframe to posteriorly train the one-class models
        output_dataframe = pd.DataFrame(list(zip(image_full_path_list, model_responses)), columns=['image_path', 'features'])
        output_dataframe.to_json('../outputs/autoencoder_features_bonafide.json')

if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    input_modalities = list(params["input_parameters"]["input_modalities"])
    predict_images(image_base_path='../../spoofing_dataset/training_real/',
                   input_modality_list=input_modalities,
                   partial_activations=params["model_parameters"]["partial_layer"])