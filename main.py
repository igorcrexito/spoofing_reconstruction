from dataset.image_dataset import ImageDataset
from model_architectures.autoencoder import Autoencoder
from PIL import Image
import numpy as np
import random
import yaml

dataset_base_path = '../spoofing_dataset/training_real/'


def plot_reconstructed_images(model_list: list, database: list, image_size: int, model_index: int):
    """
    This is just an auxiliary method that is going to be deleted in the future
    """
    number_of_images_to_be_reconstructed = 10
    for i in range(number_of_images_to_be_reconstructed):
        sample = random.randint(0, len(database) - 1)
        image = database[sample].reshape(1, image_size, image_size, 3)
        reconstructed_image = model_list[model_index].model.predict(image)
        reconstructed_image = Image.fromarray(np.uint8(reconstructed_image[0] * 255))

        reconstructed_image.show()


if __name__ == '__main__':

    print("Reading the configuration yaml the stores the executation variables")
    with open("execution_parameters.yaml", "r") as f:
        params = yaml.full_load(f)

    print("Creating a database based on the specified base path")
    image_dataset = ImageDataset(base_path=dataset_base_path,
                                 image_size=(params["application_parameters"]["image_size"],
                                             params["application_parameters"]["image_size"]),
                                 batch_size=params["model_parameters"]["batch_size"],
                                 augmentation_list=params["input_parameters"]["augmentation_list"])

    print("Creating a simple autoencoder model for each input modality")
    model_list = []
    for index, modality in enumerate(params["input_parameters"]["input_modalities"]):
        model_list.append(Autoencoder(summarize_model=False,
                                      input_dimension=params["application_parameters"]["image_size"],
                                      input_modality=modality))

        if params["application_parameters"]["application_mode"] == "train":
            print(f"Training the model for the modality {modality}")
            model_list[index].fit_model(input_data=image_dataset, validation_data=image_dataset,
                                        number_of_epochs=params["model_parameters"]["number_of_epochs"])
        else:
            print("Loading previously trained models")
            model_list[0].model.load_weights(f"trained_models/best_autoencoder_{modality}.hdf5")