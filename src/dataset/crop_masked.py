"""
crop_masked.py contains the code for preparing a Time Distributed masked data set, and loading
the prepared data set for use.
"""

import json
import os
import re

from math import floor
from random import Random
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

import cv2
from albumentations import (
    Compose, HorizontalFlip,  VerticalFlip,
    ToFloat, ShiftScaleRotate,
    )

from ..asf_typing import TimeseriesMetadataFrameKey
from ..config import NETWORK_DEMS
from ..config import TIME_STEPS, N_CHANNELS, TRAINING_LOOPS, AUGMENTATION_PROBABILITY, BATCH_SIZE, MIN_TRAINING_SAMPLES, CROP_CLASSES
from ..gdal_wrapper import gdal_open
from ..SARTimeseriesGenerator import SARTimeseriesGenerator
from .common import dataset_dir, valid_image
from keras.metrics import MeanIoU

"""Loads the training and validation timeseries metadata for a given dataset, and returns two custom keras derived
data generator iterators for that metadata, one for training data and one for validation data. 
The validation split is 10% validation, 90% training """
def load_timeseries_dataset(dataset: str) -> Tuple[SARTimeseriesGenerator]:

    # search in our dataset's root directory for json metadata files, 
    # which are generated with sample_selector.py in root project directory
    train_metadata = find_timeseries_metadata(dataset, training=True)

    # get frame keys (subdataset, frame index), which are used by the time series generator
    # to pick our time series while training each batch
    # sample size = # of time series
    # time steps = # found max frame count over all time series samples
    frame_keys, sample_size, time_steps = generate_frame_keys(train_metadata)

    # shuffle our data for validation split, with a given seed so the shuffle can be reproduced
    Random(64).shuffle(frame_keys)

    # 90% training data, 10% validation data    
    validation_split = 0.1
    split_index = floor(sample_size * validation_split)

    batch_size = BATCH_SIZE
    time_steps = TIME_STEPS
    n_classes = CROP_CLASSES
    training_p = AUGMENTATION_PROBABILITY

    print_generator_info(validation_split, time_steps, sample_size)

    # augmentations applied to training data    
    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=training_p),
        VerticalFlip(p=training_p),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, 
            rotate_limit=45, border_mode=cv2.BORDER_REFLECT_101, p=training_p),
    ])

    train_iter = SARTimeseriesGenerator(
        train_metadata, 
        time_series_frames=frame_keys[:-split_index],
        batch_size=batch_size,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=N_CHANNELS,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        n_classes=n_classes,
        dataset_directory=dataset_dir(dataset),
        shuffle=True,
        min_samples=MIN_TRAINING_SAMPLES,
        augmentations=AUGMENTATIONS_TRAIN)

    validation_iter = SARTimeseriesGenerator(
        train_metadata,
        time_series_frames=frame_keys[int(-split_index):],
        batch_size=batch_size,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=N_CHANNELS,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        n_classes=n_classes,
        dataset_directory=dataset_dir(dataset),
        shuffle=True,
        training = False)

    return train_iter, validation_iter

"""Loads the testing timeseries metadata for a given dataset, and returns a keras derived
data generator iterator for that metadata"""
def load_test_timeseries_dataset(dataset: str) -> Tuple[List[Dict], SARTimeseriesGenerator]:

    test_metadata = find_timeseries_metadata(dataset, training=False)
    frame_keys, sample_size, time_steps = generate_frame_keys(test_metadata)
    time_steps = TIME_STEPS
    batch_size=BATCH_SIZE
    n_classes=2
    test_iter = SARTimeseriesGenerator(
        time_series_metadata=test_metadata,
        time_series_frames=frame_keys,
        batch_size=batch_size,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=N_CHANNELS,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        dataset_directory=dataset_dir(dataset),
        n_classes=n_classes,
        training=False,
        shuffle=False)
    
    # we'll need the test_metadata for later when we save the predictions
    return test_metadata, test_iter

"""Scans the dataset's root directory for json metadata files, which are generated with sample_selector.py
The function returns a dictionary with a key for each metadata file it finds, each containing the associated
filepaths for testing/training data"""
def find_timeseries_metadata(dataset: str, training: bool = False) -> Dict:
    
    metadata = {}
    files = os.listdir(dataset_dir(dataset))
    
    print("Found metadata files:")

    for file in files:
        f, key = open_dataset_metadata_file(file, dataset)

        # if the file was a json file
        if key:
            print(f"\t{file}")
            if training:
                metadata[key] = f.get("train")
            else:
                metadata[key] = f.get("test")

    return metadata

''' Opens json subdataset metadata file (such as "WA_2018_metadata.json") located in a dataset's root dir 
    Params: file (json metadata file), dataset root dir name (IE: "US_SW", "US_NW")
    Returns: file subdataset prefix (In the case of the given file, "WA_2018")'''
def open_dataset_metadata_file(file: str, dataset: str):

    if file.endswith('.json'):
        with open(os.path.join(dataset_dir(dataset), file)) as json_file:
            data = json.load(json_file)
            key = file.split(".")[0]
    
            return data, key
    else:
        return None, None

"""From the metadata returned by find_timeseries_metadata 
we search for valid (non-empty)timeseries sample frame indices
in the form of ulx_####_uly_####, and their associated dataset. 

Returns 
frame_keys (a list of tuples (dataset, frame index))
sample_size (the number of timeseries)
time_steps (the maximum number of timesteps found across all timeseries)
"""
def generate_frame_keys(metadata: Dict) -> Tuple[List[TimeseriesMetadataFrameKey], int, int]:
    frame_keys = []
    time_steps=0
    sample_size=0
    total_files=0
    sub_datasets = list(metadata)

    for sub_dataset in sub_datasets:
        subset_sample_size = 0
        subset_file_count = 0

        for key in list(metadata[sub_dataset]):
            valid_files = []

            if len(metadata[sub_dataset][key]) != 0:
                time_steps = int(max(len(metadata[sub_dataset][key]), time_steps))
                subset_file_count += len(metadata[sub_dataset][key])
                subset_sample_size += 1
                frame_keys.append((sub_dataset, key))
        
        print(f"\nSubset Sample {sub_dataset} Size: {subset_sample_size} samples")
        print(f"\tsubset file count: {subset_file_count}")
        total_files+=subset_file_count
            
    sample_size = len(frame_keys)
    print("\n# of datasets:\t", len(metadata))
    print(f"\tTotal Files:\t{total_files}")
    print(f"\tCombined Sample Size:\t{sample_size} time series samples")
    print(f"\tMax Time Steps:\t{time_steps}")

    return frame_keys, int(sample_size), int(time_steps)



"""Validates image data for a vv and vh composite image."""
def validate_image(path: str, dir: str, image: str) -> bool:
    try:
        with gdal_open(os.path.join(path, dir, image)) as f:
            tile_vv_array = f.ReadAsArray()
    except FileNotFoundError:
        return False
    try:
        with gdal_open(os.path.join(path, dir, image.replace('VV', 'VH'))):
            tile_vh_array = f.ReadAsArray()
    except FileNotFoundError:
        return False
    # if not edit:
    tile_array = np.stack((tile_vh_array, tile_vv_array), axis=2)
    if not valid_image(tile_array):
        return False
    
    return True

"""Prints information related to the model"""
def print_generator_info(validation_split, time_steps, sample_size):
    split_index = floor(sample_size * validation_split)
    print("\n")
    print(f"validation Split:\t{validation_split * 100}%")
    print(f"Timesteps:\t{time_steps}")
    print(f"Base Training Samples:\t{sample_size-split_index}")
    print(f"Base Validation Samples:\t{split_index}")

    print(f"Minimum samples:\t{MIN_TRAINING_SAMPLES}")