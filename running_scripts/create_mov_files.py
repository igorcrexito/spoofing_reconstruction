from PIL import Image as im
import h5py
import numpy as np
import glob
import pandas as pd
import os
import cv2

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
        file_name = file_path.split('/')[-1][:-4]
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
            protocol_file['video_raw_name'] = protocol_file.video_name.str.split('/').apply(lambda x: x[1])
            subset = protocol_file[protocol_file['video_raw_name'] == file_name]['subset'].to_list()[0]
            subset = subset_conversion[subset]

            ## composing the output path to write the frames
            if type == 'bonafide':
                output_path = f'../../hq_wmca_rgb/{type}/{subset}/{file_name}/'
            else:
                output_path = f'../../hq_wmca_rgb/{writing_type}/{type}/{subset}/{file_name}/'

            ## reading hdf5 video file and extracting frames
            video = cv2.VideoCapture(file_path)
            while video.isOpened():
                # Read the next frame
                ret, frame = video.read()

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                frame = cv2.resize(frame, (256, 256))
                cv2.imwrite(f'{output_path}0.png', frame)
                break
        except:
            print(f'Incorrect files: {incorrect_files}. This file is not in the protocol: {file_path}')
            incorrect_files = incorrect_files + 1

    dataframe = pd.DataFrame(number_of_frames_list, columns=['number_of_frames'])
    print(dataframe.value_counts())
    __import__("IPython").embed()

if __name__ == '__main__':

    ## setting the path to the videos
    base_path = '../../raw_hq_wmca_rgb/*/*/*.mov'
    list_of_videos = glob.glob(base_path)

    ## reading the protocol file
    protocol_file = pd.read_csv('../hq_wmca_protocol/PROTOCOL-grand_test-curated.csv', sep=',')

    read_and_process_videos_hdf5(list_of_videos=list_of_videos, protocol_file=protocol_file, num_samples=1)