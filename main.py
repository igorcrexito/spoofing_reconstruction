import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os

dataset_base_path = '../spoofing_dataset/training_real/'

if __name__ == '__main__':
    print("Retrieving the file list inside the dataset folder")
    image_list = os.listdir(dataset_base_path)

