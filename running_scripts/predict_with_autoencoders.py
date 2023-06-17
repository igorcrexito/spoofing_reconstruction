import sys

import PIL
import pandas as pd


sys.path.insert(0, '..')
import tqdm
from model_architectures.autoencoder import Autoencoder
from PIL import Image, ImageOps
import numpy as np
import os
import yaml
from descriptors.bsif_descriptor import BSIFDescriptor
from descriptors.noise_descriptor import NoiseDescriptor
from descriptors.reflectance_descriptor import ReflectanceDescriptor
from descriptors.elbp_descriptor import ELBPDescriptor
import glob
import csv
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


def predict_images(image_base_path: str, input_modality_list: list, image_size: tuple = (224, 224), partial_activations: str = "", working_modality: str="bonafide", device: str='disp_1', operation: str = 'train'):
    image_list = glob.glob(image_base_path)

    ae_list = []
    bsif_descriptor_list = []
    noise_descriptor = None
    reflectance_descriptor = None
    print('Loading previously trained Autoencoder models')
    for modality in input_modality_list:
        autoencoder_model = Autoencoder(input_modality=modality,
                                        input_dimension=image_size[0],
                                        summarize_model=False,
                                        intermediate_layer=partial_activations)
        autoencoder_model.model.load_weights(f"trained_models/best_autoencoder_{modality}.hdf5")
        ae_list.append(autoencoder_model)

        ## instantiating descriptors
        if "bsif" in modality:
            bsif_descriptor = BSIFDescriptor(descriptor_name='bsif',
                                             base_path='../filters/texturefilters/',
                                             extension='.mat',
                                             filter_size=modality[5:])
            bsif_descriptor_list.append(bsif_descriptor)

        elif modality == "noise":
            noise_descriptor = NoiseDescriptor(descriptor_name="noise_descriptor")

        elif modality == 'reflectance':
            reflectance_descriptor = ReflectanceDescriptor(descriptor_name='reflectance', sigma=15)

        elif modality == 'elbp':
            elbp_descriptor = ELBPDescriptor(descriptor_name='elbp', radius=1, neighbors=8, method='default')

    model_responses = []
    image_full_path_list = []
    for index, image_path in tqdm.tqdm(enumerate(image_list)):
        image_full_path = f"{image_path}"
        image = Image.open(image_full_path).resize((image_size[0], image_size[1]))
        if partial_activations != "":
            if index == 0: print(f'This is a partial prediction, retrieving activations from: {partial_activations}')

            ae_activation = []
            for model in ae_list:
                if model.input_modality == 'rgb':
                    current_image = _normalize_image(image=np.array(image))
                    current_image = np.reshape(current_image, (1, image_size[0], image_size[1], 3))
                    prediction = model.partial_model.predict(current_image)[0]
                    ae_activation.append(prediction)

                elif 'bsif' in model.input_modality:
                    current_image = ImageOps.grayscale(image)
                    bsif_descriptor = None
                    for descriptor in bsif_descriptor_list:
                        if model.input_modality[5:] in descriptor.filter_size:
                            bsif_descriptor = descriptor
                            break

                    if bsif_descriptor is None:
                        raise ValueError("No descriptor is trained with the specified filter size")

                    current_image, _ = bsif_descriptor.compute_feature(image=np.array(current_image))
                    current_image = np.reshape(current_image, (1, image_size[0], image_size[1], 3))
                    current_image = _normalize_image(image=np.array(current_image))
                    prediction = model.partial_model.predict(current_image)[0]
                    ae_activation.append(prediction)

                elif model.input_modality == 'noise':
                    current_image = noise_descriptor.compute_feature(image=np.array(image))
                    current_image = np.reshape(current_image, (1, image_size[0], image_size[1], 3))
                    current_image = _normalize_image(image=np.array(current_image))
                    prediction = model.partial_model.predict(current_image)[0]
                    ae_activation.append(prediction)

                elif model.input_modality == 'reflectance':
                    current_image = reflectance_descriptor.compute_feature(image=np.array(image))
                    current_image = np.reshape(current_image, (1, image_size[0], image_size[1], 3))
                    current_image = _normalize_image(image=np.array(current_image))
                    prediction = model.partial_model.predict(current_image)[0]
                    ae_activation.append(prediction)

                elif model.input_modality == 'elbp':
                    current_image = elbp_descriptor.compute_feature(image=np.array(image))
                    current_image = np.reshape(current_image, (1, image_size[0], image_size[1], 3))
                    current_image = _normalize_image(image=np.array(current_image))
                    prediction = model.partial_model.predict(current_image)[0]
                    ae_activation.append(prediction)

            ae_activation = np.array(ae_activation, dtype=np.float16).flatten()
            model_responses.append(ae_activation)
            image_full_path_list.append(image_full_path)
        else:
            status_string = f'This is a full prediction, retrieving the output label obtained with the models'
            if index == 0: print(status_string)
            output = predict_image_class(image=image, autoencoder_list=ae_list, class_dictionary={1: 'rgb', 2: 'noise'})

            model_responses.append(output)
            image_full_path_list.append(image_full_path)

    ## normalizing data by column
    column_max = np.max(model_responses, axis=0)
    column_max_no_zeros = np.where(column_max == 0, 1, column_max)
    model_responses = model_responses/column_max_no_zeros

    with open(f'../outputs/autoencoder_features_{working_modality}_{device}_{operation}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in model_responses:
            writer.writerow(row)

    print("Activation file is written !")


if __name__ == '__main__':
    print("Reading the configuration yaml the stores the executation variables")
    with open("../execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    operation = params["dataset_parameters"]["execution_mode"]

    input_modalities = list(params["input_parameters"]["input_modalities"])
    for disp in ['disp_1', 'disp_2']:

        if operation == 'train':
            ## retrieving features from train dataset (bonafide)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}bonafide/{disp}/{operation}/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='bonafide',
                           device=disp,
                           operation=operation)
        else:
            ## retrieving features from train dataset (bonafide)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}bonafide/{disp}/test/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='bonafide',
                           device=disp,
                           operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack/{disp}/test/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='attack',
                           device=disp,
                           operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_cce/{disp}/test/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='attack_cce',
                           device=disp,
                           operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_hp/{disp}/test/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='attack_hp',
                           device=disp,
                           operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_print1/{disp}/test/*/*/*.jpg',
                           input_modality_list=input_modalities,
                           partial_activations=params["model_parameters"]["partial_layer"],
                           working_modality='attack_print1',
                           device=disp,
                           operation=operation)