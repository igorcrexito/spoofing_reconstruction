import os

import PIL.Image
from PIL import Image
from PIL import ImageEnhance
import numpy as np


class ImageDataset:
    def __init__(self, base_path: str, image_size: tuple, augmentation_list: list = []):
        """
        Instantiates the image dataset based on the provided image path
        """
        self.base_path = base_path
        self.image_size = image_size
        self.augmentation_list = augmentation_list

    def load_data(self):
        """
        Reads images from the specified path and apply data augmentation
        """
        image_dataset = []

        image_list = os.listdir(self.base_path)
        for image_path in image_list:
            image = Image.open(f"{self.base_path}/{image_path}").resize((self.image_size[0], self.image_size[1]))
            image_dataset.append(self._normalize_image(image))

            for augmentation in self.augmentation_list:
                image_augmented = self._apply_augmentation(image=image, augmentation=augmentation)
                image_dataset.append(self._normalize_image(image_augmented))

        return image_dataset

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
                #image[..., i] -= minval
                image[..., i] /= maxval
        # return Image.fromarray(image.astype('uint8'), 'RGBA')
        return image
