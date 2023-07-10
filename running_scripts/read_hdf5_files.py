import cv2
import h5py
import numpy as np

def read_and_process_videos_hdf5(file_path, num_samples, resize_shape=(256, 256)):

    with h5py.File(file_path, 'r') as hdf5_file:
        # Get the video dataset

        root_group = hdf5_file['/']
        group_names = list(root_group.keys())

        for group_name in group_names:
            if 'Frame_' in group_name:
                group = root_group[group_name]

                # Convert the group (frame) to an OpenCV image
                frame = np.array(group['array'])

        ## TODO this is the file read to be saved. We need to subsample this video and store some frames. Keep in mind that
        ## TODO the first slot represents the image in grayscale. The remaining channels are SWIR computations. We need to convert
        ## to PNG-like images or just array matrices

if __name__ == '__main__':

    video_folder = '../../video_samples/hdf5/file.hdf5'
    read_and_process_videos_hdf5(file_path=video_folder, num_samples=1)