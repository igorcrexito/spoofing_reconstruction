from PIL import Image as im
import h5py
import numpy as np
import glob
import pandas as pd
import os

type_id_conversion = {
    '00': 'bonafide',
    '01': 'glasses',
    '02': 'mannequin',
    '03': 'print',
    '04': 'replay',
    '05': 'rigid_mask',
    '06': 'flexible_mask',
    '07': 'paper_mask',
    '08': 'wigs',
    '09': 'tattoo',
    '10': 'makeup'
}

subset_conversion = {
    'eval': 'test',
    'dev': 'val',
    'train': 'train'
}

def read_and_process_videos_hdf5(list_of_videos, protocol_file, num_samples, resize_shape=(256, 256)):
    '''
    Reads HDF5 files and extracts the specified number of frames from the video. At the end. saves the selected frames
    '''

    incorrect_files = 0
    number_of_frames_list = []

    for file_path in list_of_videos:

        ## removing the extension and getting string parts
        reference_folder = file_path.split('/')[-2]
        file_name = file_path.split('/')[-1][:-5]
        file_name_parts = file_name.split('_')

        ## retrieving file specifications on path
        site_id = file_name_parts[0]
        session_id = file_name_parts[1]
        client_id = file_name_parts[2]
        presenter_id = file_name_parts[3]
        type_id = file_name_parts[4]
        subtype_id = file_name_parts[5]

        ## retrieving the type name based on the file name
        type = type_id_conversion[type_id]
        if type != 'bonafide':
            writing_type = 'attack'

        ## retrieving the subset to determine if the video is going to be used to train, test or validate
        try:
            subset = protocol_file[protocol_file.video_name == (reference_folder + '/' + file_name)]['subset'].to_list()[0]
            subset = subset_conversion[subset]

            ## reading hdf5 video file and extracting frames
            with h5py.File(file_path, 'r') as hdf5_file:
                root_group = hdf5_file['/']

                ## getting the list of keys that exist on the group. In other words, these are the frames
                group_names = list(root_group.keys())
                group_names = [x for x in group_names if 'Frame_' in x]

                number_of_frames_list.append(len(group_names))
                ## retrieving first frame of the video for every channel
                group = root_group[group_names[0]]

                # Convert the group (frame) to an OpenCV image
                frame = np.array(group['array'], dtype='uint8')

                ## composing the output path to write the frames
                if type == 'bonafide':
                    output_path = f'../../hq_wmca/{type}/{subset}/{file_name}/'
                else:
                    output_path = f'../../hq_wmca/{writing_type}/{type}/{subset}/{file_name}/'

                for channel in range(0, frame.shape[0]):
                    channel_image = im.fromarray(frame[channel]).resize((256, 256))

                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

                    writing_path = output_path + str(channel) + '.png'
                    channel_image.save(writing_path)
        except:
            print(f'Incorrect files: {incorrect_files}. This file is not in the protocol: {file_path}')
            incorrect_files = incorrect_files + 1

    dataframe = pd.DataFrame(number_of_frames_list, columns=['number_of_frames'])
    print(dataframe.value_counts())
    __import__("IPython").embed()


if __name__ == '__main__':

    ## setting the path to the videos
    base_path = '../../raw_hq_wmca/*/*.hdf5'
    list_of_videos = glob.glob(base_path)

    ## reading the protocol file
    protocol_file = pd.read_csv('../hq_wmca_protocol/PROTOCOL-grand_test-curated.csv', sep=',')

    read_and_process_videos_hdf5(list_of_videos=list_of_videos, protocol_file=protocol_file, num_samples=1)