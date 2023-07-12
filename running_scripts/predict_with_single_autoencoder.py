import sys
import PIL

sys.path.insert(0, '..')
import tqdm
from custom_losses.custom_losses import focal_loss_for_regression
from model_architectures.single_autoencoder import SingleAutoencoder
from PIL import Image, ImageOps
import numpy as np
import yaml
import glob
import csv
from vit_keras import vit
import multiprocessing as mp

autoencoders_base_path = 'trained_models/'


def _normalize_image(image: PIL.Image.Image, number_of_channels: int = 3):
    image = np.array(image).astype('float')

    for i in range(number_of_channels):
        minval = image[:, :, i].min()

        if minval <= 0:
            image[:, :, i] += abs(minval)
        else:
            image[:, :, i] -= minval

        maxval = image[:, :, i].max()
        image[:, :, i] /= maxval
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


def predict_images(image_base_path: str, image_size: tuple = (224, 224),
                   partial_activations: str = "", working_modality: str = "bonafide", device: str = 'disp_1',
                   operation: str = 'train'):

    print("Retrieving the image list")
    image_list = glob.glob(image_base_path)

    print('Loading previously trained single Autoencoder model')
    autoencoder_model = SingleAutoencoder(input_dimension=image_size[0],
                                          summarize_model=False,
                                          intermediate_layer=partial_activations)

    autoencoder_model.model.load_weights(f'trained_models/{params["application_parameters"]["dataset"]}/best_autoencoder_single_autoencoder.hdf5')

    print("Iterating over images")
    model_responses = []

    for index, image_path in tqdm.tqdm(enumerate(image_list)):
        image_full_path = f"{image_path}"
        image = Image.open(image_full_path).resize((image_size[0], image_size[1]))

        if partial_activations != "":
            if index == 0: print(f'This is a partial prediction, retrieving activations from: {partial_activations}')

            ## rgb input
            rgb_image = _normalize_image(image=vit.preprocess_inputs(np.array(image)))

            #prediction = autoencoder_model.partial_model.predict([bsif_image, reflectance_image, elbp_image])[0]
            output_image = np.reshape(rgb_image, (1, image.width, image.height, 3))
            prediction = autoencoder_model.partial_model.predict([output_image])
            prediction = np.array(prediction, dtype=np.float16).flatten()

            ## storing the predictions into a list
            model_responses.append(prediction)
        else:
            ## this is going to be implemented soon - TODO
            pass

    ## normalizing data by column
    column_max = np.max(model_responses, axis=0)
    column_max_no_zeros = np.where(column_max == 0, 1, column_max)
    model_responses = model_responses / column_max_no_zeros

    with open(f'../outputs/{params["application_parameters"]["dataset"]}/autoencoder_features_single_model_{working_modality}_{device}_{operation}.csv', mode='w',
              newline='') as file:
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
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}bonafide/{disp}/{operation}/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='bonafide',
                device=disp,
                operation=operation)
        elif operation == 'test':
            ## retrieving features from train dataset (bonafide)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}bonafide/{disp}/test/*/*/*.jpg',
                               partial_activations=params["model_parameters"]["partial_layer"],
                               image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                               working_modality='bonafide',
                               device=disp,
                               operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack/{disp}/test/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack',
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_cce/{disp}/test/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack_cce',
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_hp/{disp}/test/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                working_modality='attack_hp',
                image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_print1/{disp}/test/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack_print1',
                device=disp,
                operation=operation)
        else:
            ## retrieving features from train dataset (bonafide)
            predict_images(image_base_path=f'{params["dataset_parameters"]["base_path"]}bonafide/{disp}/val/*/*/*.jpg',
                           partial_activations=params["model_parameters"]["partial_layer"],
                           image_size=(params["application_parameters"]["image_size"],
                                       params["application_parameters"]["image_size"]),
                           working_modality='bonafide',
                           device=disp,
                           operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack/{disp}/val/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(
                params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack',
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_cce/{disp}/val/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(
                params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack_cce',
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_hp/{disp}/val/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                working_modality='attack_hp',
                image_size=(
                params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                device=disp,
                operation=operation)

            ## retrieving features from test dataset (attack)
            predict_images(
                image_base_path=f'{params["dataset_parameters"]["base_path"]}attack/attack_print1/{disp}/val/*/*/*.jpg',
                partial_activations=params["model_parameters"]["partial_layer"],
                image_size=(
                params["application_parameters"]["image_size"], params["application_parameters"]["image_size"]),
                working_modality='attack_print1',
                device=disp,
                operation=operation)