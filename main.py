from dataset.image_dataset import ImageDataset
from model_architectures.autoencoder import Autoencoder
from PIL import Image
import numpy as np
import random

dataset_base_path = '../spoofing_dataset/training_real/'
augmentation_list = ["flip", "zoom", "rotate"]
input_modalities = ['rgb']
model_list = []

## training parameters
number_of_epochs = 10
batch_size = 50
image_size = 112

if __name__ == '__main__':

    print("Creating a database based on the specified base path")
    image_dataset = ImageDataset(base_path=dataset_base_path, image_size=(image_size, image_size), augmentation_list=augmentation_list)

    print("Loading the images and applying the data augmentation operations")
    database = np.array(image_dataset.load_data(), dtype='float16')

    print("Creating a simple autoencoder model for each input modality")
    for modality in input_modalities:
        model_list.append(Autoencoder(summarize_model=True, input_dimension=image_size, input_modality=modality))
        model_list[0].fit_model(input_data=database, validation_data=database, number_of_epochs=number_of_epochs, batch_size=batch_size)


    ## just testing the reconstruction for some images
    #number_of_images_to_be_reconstructed = 10
    #for i in range(number_of_images_to_be_reconstructed):
    #    sample = random.randint(0, len(database)-1)
    #    image = database[sample].reshape(1, image_size, image_size, 3)
    #    reconstructed_image = model_list[0].model.predict(image)
    #    reconstructed_image = Image.fromarray(np.uint8(reconstructed_image[0]*255))

    #    reconstructed_image.show()