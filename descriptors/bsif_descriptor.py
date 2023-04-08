import pandas as pd

from descriptors.image_descriptor import ImageDescriptor
import numpy as np
import scipy
import math
from scipy import signal
import glob
from PIL import Image, ImageOps
import tqdm


class BSIFDescriptor(ImageDescriptor):

    def __init__(self, descriptor_name: str, base_path: str = 'filters/texturefilters/', extension: str = "*.mat",
                 compute_histogram: bool = False, image_size: tuple = (224, 224)):
        super().__init__(descriptor_name)
        self.base_path = base_path
        self.extension = extension
        self.compute_histogram = compute_histogram
        self.image_size = image_size

    def compute_feature(self, image: np.ndarray):

        histogram_list = []
        encoded_image_list = []

        ## retrieving all filters from the filter folder
        filter_list = glob.glob(f"{self.base_path}{self.extension}")
        filter_list.sort()

        for texture_filter_path in filter_list:
            f = scipy.io.loadmat(texture_filter_path)
            texturefilters = f.get('ICAtextureFilters')

            # Initialize
            numScl = np.shape(texturefilters)[2]
            codeImg = np.ones(np.shape(image))

            # Make spatial coordinates for sliding window
            r = int(math.floor(np.shape(texturefilters)[0] / 2))

            # Wrap image (increase image size according to maximum filter radius by wrapping around)
            upimg = image[0:r, :]
            btimg = image[-r:, :]
            lfimg = image[:, 0:r]
            rtimg = image[:, -r:]
            cr11 = image[0:r, 0:r]
            cr12 = image[0:r, -r:]
            cr21 = image[-r:, 0:r]
            cr22 = image[-r:, -r:]

            imgWrap = np.vstack(
                (np.hstack((cr22, btimg, cr21)), np.hstack((rtimg, image, lfimg)), np.hstack((cr12, upimg, cr11))))

            # Loop over scales
            for i in range(numScl):
                tmp = texturefilters[:, :, numScl - i - 1]
                ci = signal.convolve2d(imgWrap, np.rot90(tmp, 2), mode='valid')
                t = np.multiply(np.double(ci > 0), 2 ** i)
                codeImg = codeImg + t

            if self.compute_histogram:
                hist_bsif = np.histogram(codeImg.ravel(), bins=np.arange(1, (2 ** numScl) + 2))
                hist_bsif = hist_bsif[0]
                # normalize the histogram
                hist_bsif = hist_bsif / (hist_bsif.sum() + 1e-7)
                histogram_list.append(hist_bsif)

            encoded_image_list.append(codeImg)

        return encoded_image_list, histogram_list

    def bsif_precomputation(self, dataset_path: str, extension: str = '*.jpg'):
        """
        Function to pre-compute the bsif descriptor due to the high cost
        """
        ## retrieving the filters and images
        filter_list = glob.glob(f"{self.base_path}{self.extension}")
        filter_list.sort()

        image_list = glob.glob(f"{dataset_path}{extension}")

        ## creating columns into a dataframe based on the number of filters
        columns = [x.split('/')[-1] for x in filter_list]
        output_dataframe = pd.DataFrame(columns=columns)

        row_index = 0
        for image_path in tqdm.tqdm(image_list):
            image = Image.open(image_path).resize((self.image_size[0], self.image_size[1]))
            image = np.array(ImageOps.grayscale(image))
            bsif_descriptor = BSIFDescriptor(descriptor_name="bsif")

            computed_images, _ = bsif_descriptor.compute_feature(image)
            for index, feature in enumerate(computed_images):
                output_dataframe.loc[row_index, columns[index]] = np.uint16(feature.flatten())
            row_index = row_index + 1

        return output_dataframe
