from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
import tensorflow as tf


class Autoencoder():

    def __init__(self, input_modality: str, input_dimension: int, summarize_model: bool):
        self.input_modality = input_modality
        self._input_dimension = input_dimension
        self.model = self._create_model(summarize_model=summarize_model)

    def _create_model(self, summarize_model: bool = False):
        input = Input(shape=(self._input_dimension, self._input_dimension, 3), name='main_input')

        # encoding input into a higher level representation - 1st group
        encoding = Conv2D(32, (3, 3), activation='relu', padding='same')(input)
        encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(encoding)

        encoding = Conv2D(64, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(encoding)

        encoding = Conv2D(128, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(256, (3, 3), activation='relu', padding='same')(encoding)
        encoding = Conv2D(256, (3, 3), activation='relu', padding='same')(encoding)
        encoding = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(encoding)

        # decoding step -> getting back to original representation
        decoding = Conv2D(256, (3, 3), activation='relu', padding='same')(encoding)
        decoding = Conv2D(256, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up1')(decoding)

        decoding = Conv2D(128, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(128, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(64, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up2')(decoding)

        decoding = Conv2D(64, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(64, (3, 3), activation='relu', padding='same')(decoding)
        decoding = Conv2D(32, (3, 3), activation='relu', padding='same')(decoding)
        decoding = UpSampling2D(size=(2, 2), name='up3')(decoding)

        output = Conv2D(3, (3, 3), activation='relu', padding='same')(decoding)

        autoencoder = Model([input], [output])
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        if summarize_model:
            autoencoder.summary()

        return autoencoder

    def fit_model(self, input_data: np.ndarray, validation_data: np.ndarray, number_of_epochs: int):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'trained_models/best_autoencoder_{self.input_modality}.hdf5',
                                                        monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        self.model.fit_generator(epochs=number_of_epochs,
                       shuffle=True,
                       generator=input_data,
                       validation_data=validation_data,
                       use_multiprocessing=True,
                       workers=6,
                       callbacks=[checkpoint])
