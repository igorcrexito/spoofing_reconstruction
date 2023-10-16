from descriptors.image_descriptor import ImageDescriptor
import numpy as np
import cv2


class SpectralDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, sigma: float, kernel_dimension: int):
        super().__init__(descriptor_name)
        self.sigma = sigma
        self.kernel_dimension = kernel_dimension

    def compute_feature(self, image: np.ndarray):
        kernel_size = (self.kernel_dimension, self.kernel_dimension)

        ## computing the residual image
        blurred_image = cv2.GaussianBlur(image, kernel_size, self.sigma)
        residual_image = image - blurred_image

        ## computing the Fast Fourier Transform of the image
        fft_image = cv2.dft(np.float32(residual_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        fft_shifted = np.fft.fftshift(fft_image)

        #magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shifted[:, :, 0], fft_shifted[:, :, 1]))

        #return np.reshape(magnitude_spectrum, (256, 256, 1))
        return fft_shifted