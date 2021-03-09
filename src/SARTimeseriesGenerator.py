# Basic class organization based on Shervine Amidi's implementation,
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import os
import random
import re
from src.CDLFrameData import CDLFrameData 
from typing import Dict, List, Optional, Tuple
from src.config import TIME_STEPS, MIN_TIME_STEPS, NETWORK_DEMS
import keras
import numpy as np
import cv2
from albumentations import (
    Compose, ToFloat
)
from src.gdal_wrapper import gdal_open
from tensorflow import convert_to_tensor

from .asf_typing import TimeseriesMetadataFrameKey

"""Takes metadata dictionary and list of timeseriesMetadataFrameKeys (dataset name, key) to access said sample."""
class SARTimeseriesGenerator(keras.utils.Sequence):
    def __init__(self, time_series_metadata: Dict, time_series_frames: List[TimeseriesMetadataFrameKey], batch_size=32, dim=(NETWORK_DEMS,NETWORK_DEMS), 
    time_steps=TIME_STEPS, min_time_steps=MIN_TIME_STEPS, n_channels=2, output_dim=(NETWORK_DEMS, NETWORK_DEMS), output_channels=1, 
    n_classes=7, shuffle=True, dataset_directory="", clip_range: Optional[Tuple[float, float]] = None, training = True, augmentations=Compose([ToFloat(max_value=255, p=0.0),
        ]), min_samples=20000):
        self.class_mode = 'categorical'
        self.list_IDs = time_series_metadata
        self.frame_data = time_series_frames        
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.dim = dim
        self.training = training
        self.time_steps = time_steps
        self.min_time_steps = min_time_steps
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.clip_range = clip_range
        
        self.augment = augmentations
        self.metadata = []
        
        self.init = False

        if training:
            self.__meet_min_steps()
            while len(self.frame_data) < min_samples:
                self.frame_data.extend(random.sample(self.frame_data, min(min_samples - len(self.frame_data), len(self.frame_data))))

        # print("PRE INIT:\n", len(self.frame_data))
        self.frame_data = [CDLFrameData(sample[0], sample[1], dataset_directory) for sample in self.frame_data]
        # print(len(self.frame_data), "\n")
        self.on_epoch_end() 

    def __len__(self):
        #Returns amount of batches per epochs
        return int(np.floor(len(self.frame_data) / self.batch_size))

    def __getitem__(self, index):
        # get indices in current batch
        indexes = self.indexes[index*self.batch_size : (index+1) * self.batch_size]
        # indexes = self.index
        # get list of IDs
        frame_data_temp = [self.frame_data[k] for k in indexes]
        
        X, y = self.__data_generation(frame_data_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.frame_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, frame_data_temp):
        if len(frame_data_temp) == 0:
            print("FRAME DATA EMPTY")

        sample_metadata = []

        # (samples, timesteps, width, height, channels)
        X = np.zeros((self.batch_size, *self.dim, self.n_channels * self.time_steps), dtype=np.float32)
        y = []

        if self.n_classes > 2:
            y = np.zeros((self.batch_size, *self.output_dim, self.n_classes), dtype=np.float32)
        else:
            y = np.zeros((self.batch_size, *self.output_dim, 1), dtype=np.float32)

        # sample_subset_prefix: "WA_2018", "OR_2017"
        #frame_number: "ulx_0_uly_0"
        for sample_idx, sample_key_data in enumerate(frame_data_temp):
            
            frame_number = sample_key_data.get_frame_index_key()
            sample_subset_prefix = sample_key_data.get_sub_dataset_name()

            if self.training:
                X[sample_idx,], y[sample_idx] = sample_key_data.load_timeseries_frame_data(self.list_IDs[sample_subset_prefix][frame_number], self.dim, self.n_channels, self.time_steps, self.augment)
            else:
                X[sample_idx,], y[sample_idx] = sample_key_data.load_timeseries_frame_data(self.list_IDs[sample_subset_prefix][frame_number], self.dim, self.n_channels, self.time_steps)             
                sample_metadata.append(sample_key_data.get_metadata_paths())

        #keep track of testing data
        if not self.training:
            self.__setBatchMetadata(sample_metadata)

        return np.nan_to_num(X, nan=0, copy=False), np.nan_to_num(y, nan=0, copy=False)


    # Non-keras.utils.Sequence functions

    """Remove all samples that don't meet the minimum amount of timesteps"""
    def __meet_min_steps(self):
        # old_sample_size = sum([len(self.list_IDs[k]) for k in self.list_IDs])
        old_sample_size = len(self.frame_data)

        for idx, sample_prefix_and_frame in enumerate(self.frame_data):
            sample_subset_prefix, frame_number = sample_prefix_and_frame
            frame = self.list_IDs[sample_subset_prefix][frame_number]
            if len(frame) < self.min_time_steps:
                self.list_IDs[sample_subset_prefix].pop(frame_number)
                self.frame_data[idx] = None
        
        self.frame_data = [x for x in self.frame_data if x != None]

        print(f"Samples: {old_sample_size}")
        print(f"Samples with at least {self.min_time_steps} time steps:\t {len(self.frame_data)}")
           

    '''accessor to get metadata, ordered by input, contains strings of file paths and their 
    crop mask in the order they're fed to the model'''
    def getBatchMetadata(self) -> List[Tuple[List[Tuple[str, str]], str]]:
        return self.metadata

    '''when each batch begins we record what files are passed to the model'''
    def __setBatchMetadata(self, batch: Tuple[List[Tuple[str, str]], str]):
        # print(len(self.metadata), "\n")
        # print(batch, "\n")
        # for data in self.metadata:
        #     for sample in data:
        #         if data
            # if batch 
        # if batch not in self.metadata:
        if not self.init:
            self.init = True
        else:
            self.metadata.append(batch)


# Classes:
# Corn: 229 208 18, 255
# Cotton: 203, 79, 128, 255
# Rice: 42, 116, 80, 255
# Sorghum: 255, 148, 33, 255
# Soybeans: 89, 210, 176, 255
# Peanuts: 83, 43, 116, 255
# Sprint Wheat: 205, 139, 25, 255
# Winter Wheat: 205, 173, 117, 255
# Dbl Crop WinWht/Corn: 205, 194, 175, 255
# Alfalfa: 182, 89, 210, 255
