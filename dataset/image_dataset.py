import cv2
import os


class ImageDataset:
    def __init__(self, base_path: str, batch_size: int):
        self.base_path = base_path
        self.batch_size = batch_size

    def augment_data(self, augmentation_type: str, input_image):
        pass

    def load_data(self):
        pass
