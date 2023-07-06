import os
import PIL.Image
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

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
        return image_batch[:, :, :, 0:3], image_batch[:, :, :, 3:]
        # return [image_batch[:, :, :, 0:1], image_batch[:, :, :, 1:4], image_batch[:, :, :, 4:5]], image_batch

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
        noise_descriptor = NoiseDescriptor(descriptor_name="noise_descriptor")

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
            # Store sample
            original_image = Image.open(f"{image_path}").resize((self.image_size[0], self.image_size[1]))

            ## list storing images and augmentated images
            augmentation_images = [original_image]
            for augmentation in self.augmentation_list:
                augmentation_images.append(self._apply_augmentation(image=original_image, augmentation=augmentation))

            for image in augmentation_images:
                output_image = np.ones((image.width, image.height, 10),
                                       dtype=np.float16)  ## 8 is the number of output dimensions due to features

                ## APPENDING RGB IMAGE
                output_image[:, :, 0:3] = self._normalize_image(vit.preprocess_inputs(np.array(image)))

                ## APPENDING NOISE IMAGE
                # current_image = noise_descriptor.compute_feature(image=np.array(image))
                # output_image[:, :, 3:6] = self._normalize_image(current_image)

                ### APPENDING BSIF IMAGE
                current_image = ImageOps.grayscale(image)
                current_image, _ = bsif_descriptor7x7.compute_feature(image=np.array(current_image))
                output_image[:, :, 3:4] = np.reshape(self._normalize_image(current_image, number_of_channels=1)[:, :, 0],
                                                     (image.width, image.height, 1))
                current_image = ImageOps.grayscale(image)
                current_image, _ = bsif_descriptor5x5.compute_feature(image=np.array(current_image))
                output_image[:, :, 4:5] = np.reshape(self._normalize_image(current_image, number_of_channels=1)[:, :, 0],
                                                     (image.width, image.height, 1))

                current_image = ImageOps.grayscale(image)
                current_image, _ = bsif_descriptor3x3.compute_feature(image=np.array(current_image))
                output_image[:, :, 5:6] = np.reshape(self._normalize_image(current_image, number_of_channels=1)[:, :, 0],
                                                     (image.width, image.height, 1))

                ### GENERATING DATA FOR REFLECTANCE INPUT MODALITY ###
                current_image = reflectance_descriptor.compute_feature(image=np.array(image))
                output_image[:, :, 6:9] = self._normalize_image(current_image)

                ### GENERATING DATA FOR ELBP INPUT MODALITY ###
                current_image = eblp_descriptor.compute_feature(image=np.array(image))
                output_image[:, :, 9:10] = np.reshape(self._normalize_image(current_image, number_of_channels=1)[:, :, 0],
                                                     (image.width, image.height, 1))
                X.append(output_image)

        return np.array(X, dtype=np.float16)

    def _apply_augmentation(self, image: PIL.Image.Image, augmentation: str):
        """
        Apply augmentation depending on the provided string
        """

        if augmentation == 'flip':
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif augmentation == 'zoom':
            height = image.size[0]
            width = image.size[1]
            return image.crop(
                (int(height * 0.1), int(width * 0.1), height - (int(height * 0.1)), width - (int(width * 0.1)))).resize(
                (height, width))
        elif augmentation == 'rotate':
            return image.rotate(10, expand=False)


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
