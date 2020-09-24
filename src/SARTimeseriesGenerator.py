# Basic class organization based on Shervine Amidi's implementation,
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import os
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, time_series_metadata: Dict, time_series_frames: List[TimeseriesMetadataFrameKey], batch_size=32, dim=(512, 512), 
    time_steps=1, n_channels=2, output_dim=(512, 512), output_channels=1, 
    n_classes=3, shuffle=True, dataset_directory="", clip_range: Optional[Tuple[float, float]] = None, training = True, augmentations=Compose([ToFloat(max_value=255)
        ])):
        self.list_IDs = time_series_metadata
        self.frame_data = time_series_frames
        self.dataset_directory = dataset_directory
        self.batch_size = batch_size
        self.dim = dim
        self.training = training
        self.time_steps = time_steps
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.clip_range = clip_range
        self.metadata = []
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        #Returns amount of batches per epochs
        return int(np.floor(len(self.frame_data) / self.batch_size))

    def __getitem__(self, index):
        # get indices in current batch
        indexes = self.indexes[index*self.batch_size : (index+1) * self.batch_size]
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
        self.setBatchMetadata(frame_data_temp)

        # (samples, timesteps, width, height, channels)
        X = np.zeros((self.batch_size, self.time_steps, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size, 1, *self.output_dim, self.output_channels), dtype=np.float32)

        #frame numbers are in the "ulx_0_uly_0" format
        for sample_idx, (subset_sample, frame_number) in enumerate(frame_data_temp):
            time_series_stack = []
            time_series_mask = []
            
            time_series = self.list_IDs[subset_sample][frame_number]
            time_series.sort()
            for tileVH, tileVV in zip(time_series[0::2], time_series[1::2]):
                try:
                    with gdal_open(os.path.join(self.dataset_directory,tileVH)) as f:
                        vh = f.ReadAsArray()
                except FileNotFoundError:
                    continue
                try:
                    with gdal_open(os.path.join(self.dataset_directory,tileVV)) as f:
                        vv = f.ReadAsArray()
                except FileNotFoundError:
                    continue

                tile_array = np.stack((vh, vv), axis=2).astype('float32')
                tile_array = (tile_array - np.min(tile_array))/np.ptp(tile_array)
                if self.clip_range:
                    min_, max_ = self.clip_range
                    np.clip(X, min_, max_, out=X)
                
                time_series_stack.append(tile_array)
            
            if len(time_series_stack) != 0:

                # if we end up with a stack with less than the set amount of timesteps, 
                # append existing elements until we get enough timesteps

                # We have three options to work around this problem of variable timesteps we either
                    # Removing rows with missing values.
                    # Mark and learn missing values.
                    # Mask and learn without missing values.
                if len(time_series_stack) < self.time_steps:
                    idx = len(time_series_stack)
                    temp = time_series_stack[0]
                    while(idx != self.time_steps):
                        time_series_stack.append(temp)
                        idx+=1

                #convert list of vv vh composites to numpy array
                x_stack = np.stack(time_series_stack, axis=0).astype('float32')
                x_stack = np.stack([self.augment(image=x)["image"] for x in x_stack], axis=0)
                subset_name = f"{'_'.join(subset_sample.split('_')[:-1])}"
                subset_mask_dir_name = f"{subset_name}_masks"
                file_name = f"CDL_{subset_name}_mask_{frame_number}.tif"
                mask = 0
                try:
                    if self.training:
                        with gdal_open(os.path.join(self.dataset_directory, "train", subset_mask_dir_name, file_name)) as f:
                            mask = f.ReadAsArray()
                    else:
                        with gdal_open(os.path.join(self.dataset_directory, "test", subset_mask_dir_name, file_name)) as f:
                            mask = f.ReadAsArray()
                except FileNotFoundError:
                    continue

                # mask_array = np.array(mask).astype('float32').reshape(512, 512, 1) / 255.0
                mask_array = np.array(mask).astype('bool').reshape(self.output_dim[0], self.output_dim[1], 1)

                X[sample_idx,] = x_stack
                y[sample_idx,] = mask_array
        
                # X = np.stack([self.augment(image=x)["image"] for x in X], axis=0)
        # return np.nan_to_num(X, nan=-1, copy=False), keras.utils.to_categorical(np.nan_to_num(y, nan=-1, copy=False), self.n_classes)
        return convert_to_tensor(np.nan_to_num(X, nan=0, copy=False)), convert_to_tensor(np.nan_to_num(y, nan=0, copy=False))


    # Non-keras.utils.Sequence functions

    # accessor to get metadata, ordered by input, contains strings of file paths and their 
    # crop mask in the order they're fed to the model
    def getBatchMetadata(self) -> List[Tuple[List[Tuple[str, str]], str]]:
        return self.metadata

    # when each batch begins we record what files are passed to the model
    def setBatchMetadata(self, batch: Tuple[List[Tuple[str, str]], str]):
        self.metadata.append(batch)
