import os
import PIL.Image
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras
from typing import List
from descriptors.bsif_descriptor import BSIFDescriptor
from descriptors.elbp_descriptor import ELBPDescriptor
from descriptors.noise_descriptor import NoiseDescriptor
from descriptors.reflectance_descriptor import ReflectanceDescriptor
from vit_keras import vit
import glob


class SingleAutoencoderImageDataset(keras.utils.Sequence):
    def __init__(self, base_path: str, image_size: tuple, batch_size: int,
                 augmentation_list: list = [], shuffle: bool = True):
        """
        Instantiates the image dataset based on the provided image path
        """
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentation_list = augmentation_list
        self.shuffle = shuffle
        self.list_of_images = glob.glob(self.base_path)

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_of_images) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_of_images_temp = [self.list_of_images[k] for k in indexes]

        # Generate data
        image_batch = self.__data_generation(list_of_images_temp)

        ## returning each image separately
        return image_batch[:, :, :, 0:43], image_batch[:, :, :, 43:48]

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_of_images))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_of_images_temp):
        """
        Generates data containing batch_size samples
        """
        X = []  ## input data

        ## instantiating descriptors
        bsif_descriptor7x7 = BSIFDescriptor(descriptor_name='bsif',
                                            base_path='../filters/texturefilters/',
                                            extension='.mat',
                                            filter_size='7x7_6')

        bsif_descriptor5x5 = BSIFDescriptor(descriptor_name='bsif',
                                            base_path='../filters/texturefilters/',
                                            extension='.mat',
                                            filter_size='5x5_6')

        bsif_descriptor3x3 = BSIFDescriptor(descriptor_name='bsif',
                                            base_path='../filters/texturefilters/',
                                            extension='.mat',
                                            filter_size='3x3_6')

        reflectance_descriptor = ReflectanceDescriptor(descriptor_name='reflectance', sigma=125)

        eblp_descriptor = ELBPDescriptor(descriptor_name='elbp', radius=1, neighbors=8, method='default')

        for i, image_path in enumerate(list_of_images_temp):

            image_struct = []

            # Store sample
            for channel in range(0, 43):
                image = Image.open(f"{image_path}0.png").resize((self.image_size[0], self.image_size[1]))
                image_struct.append(image)

            ## list storing images and augmentated images
            augmentation_images = [image_struct]
            for augmentation in self.augmentation_list:
                augmentation_images.append(
                    self._apply_augmentation(image_channels=image_struct, augmentation=augmentation))

            for image in augmentation_images:
                image = np.stack(image, axis=-1)
                output_image = np.ones((self.image_size[0], self.image_size[1], 48),
                                       dtype=np.float16)  ## 43 + BSIFs (3) + Reflectance (1) + LBP (1)
                ## is the number of output dimensions due to features

                ## APPENDING RGB IMAGE
                output_image[:, :, :43] = self._normalize_image(image=image[:, :, :43], number_of_channels=43)

                ### APPENDING BSIF IMAGE
                original_grayscale = image[:, :, 0]

                current_image, _ = bsif_descriptor7x7.compute_feature(image=original_grayscale)
                current_image = np.reshape(current_image[:, :, 0], (self.image_size[0], self.image_size[1], 1))
                output_image[:, :, 43:44] = self._normalize_image(current_image, number_of_channels=1)

                current_image, _ = bsif_descriptor5x5.compute_feature(image=original_grayscale)
                current_image = np.reshape(current_image[:, :, 0], (self.image_size[0], self.image_size[1], 1))
                output_image[:, :, 44:45] = self._normalize_image(current_image, number_of_channels=1)

                current_image, _ = bsif_descriptor3x3.compute_feature(image=original_grayscale)
                current_image = np.reshape(current_image[:, :, 0], (self.image_size[0], self.image_size[1], 1))
                output_image[:, :, 45:46] = self._normalize_image(current_image, number_of_channels=1)

                ### GENERATING DATA FOR REFLECTANCE INPUT MODALITY ###
                current_image = reflectance_descriptor.compute_feature(image=original_grayscale, num_channels=1)
                current_image = np.reshape(current_image, (self.image_size[0], self.image_size[1], 1))
                output_image[:, :, 46:47] = self._normalize_image(current_image, number_of_channels=1)

                current_image = eblp_descriptor.compute_feature(image=original_grayscale)
                output_image[:, :, 47:48] = self._normalize_image(current_image, number_of_channels=1)

                X.append(output_image)

        return np.array(X, dtype=np.float16)

    def _apply_augmentation(self, image_channels: List, augmentation: str):
        """
        Apply augmentation depending on the provided string
        """
        if augmentation == 'flip':
            for index, channel in enumerate(image_channels):
                image_channels[index] = channel.transpose(Image.FLIP_LEFT_RIGHT)

        return image_channels

    def _normalize_image(self, image: PIL.Image.Image, number_of_channels: int = 3):
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
