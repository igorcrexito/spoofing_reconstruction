import os
import PIL.Image
from PIL import Image
import numpy as np
from tensorflow import keras


class ImageDataset(keras.utils.Sequence):
    def __init__(self, base_path: str, image_size: tuple, batch_size: int, augmentation_list: list = [], shuffle: bool = True):
        """
        Instantiates the image dataset based on the provided image path
        """
        self.base_path = base_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentation_list = augmentation_list
        self.shuffle = shuffle
        self.list_of_images = os.listdir(self.base_path)

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

        return image_batch, image_batch

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_of_images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_of_images_temp):
        """
        Generates data containing batch_size samples
        """
        X = []
        # Generate data

        for i, image_path in enumerate(list_of_images_temp):
            # Store sample
            image = Image.open(f"{self.base_path}/{image_path}").resize((self.image_size[0], self.image_size[1]))
            X.append(self._normalize_image(image))

            for augmentation in self.augmentation_list:
                image_augmented = self._apply_augmentation(image=image, augmentation=augmentation)
                X.append(self._normalize_image(image_augmented))

        return np.array(X)

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

    def _normalize_image(self, image: PIL.Image.Image):
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
