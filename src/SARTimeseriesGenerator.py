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
        self.__meet_min_steps()

        # print(f"{len()}"")
        if training:
            while len(self.frame_data) < min_samples:
                self.frame_data.extend(random.sample(self.frame_data, min(min_samples - len(self.frame_data), len(self.frame_data))))
        print("key data pre:", self.frame_data[0])
        self.frame_data = [CDLFrameData(sample[0], sample[1], dataset_directory) for sample in self.frame_data]
        print("key data post:", self.frame_data[0].get_frame_index_key(), self.frame_data[0].get_dataset_name())
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.clip_range = clip_range
        self.metadata = []
        self.augment = augmentations
        self.on_epoch_end()

    """Remove all samples that don't meet the minimum amount of timesteps"""
    def __meet_min_steps(self):
        # old_sample_size = sum([len(self.list_IDs[k]) for k in self.list_IDs])
        old_sample_size = len(self.frame_data)

        for idx, sample_prefix_and_frame in enumerate(self.frame_data):
            sample_subset_prefix, frame_number = sample_prefix_and_frame
            frame = self.list_IDs[sample_subset_prefix][frame_number]
            if len(frame) < self.min_time_steps:
                # if idx < 10:
                #     print(f"{len(frame)}\n{frame}")
                self.list_IDs[sample_subset_prefix].pop(frame_number)
                self.frame_data[idx] = None
        
        self.frame_data = [x for x in self.frame_data if x != None]
            # if sample_idx == 0:
                # print(len(frame))
        # if self.training:
        print(f"Samples: {old_sample_size}")
        print(f"Samples with at least {self.min_time_steps} time steps:\t {len(self.frame_data)}")
            

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
            sample_subset_prefix = sample_key_data.get_dataset_name()

            if self.training:
                X[sample_idx,], y[sample_idx] = sample_key_data.load_timeseries_frame_data(self.list_IDs[sample_subset_prefix][frame_number], self.dim, self.n_channels, self.time_steps, self.augment)
            else:
                X[sample_idx,], y[sample_idx] = sample_key_data.load_timeseries_frame_data(self.list_IDs[sample_subset_prefix][frame_number], self.dim, self.n_channels, self.time_steps)             
                sample_metadata.append(sample_key_data.get_metadata_paths())
        # for sample_idx, (sample_subset_prefix, frame_number) in enumerate(frame_data_temp):
        #     time_series_stack = np.zeros((*self.dim, self.n_channels*self.time_steps), dtype=np.float32)
        #     time_step_idy = 0

        #     random_selection = self.__random_frame_sample(self.list_IDs[sample_subset_prefix][frame_number])

        #     # Ignore S1B / S1A prefix in sorting
        #     random_selection.sort(key=lambda pair: pair[0].split("_")[1:])
            
        #     if not self.training:
        #         sample_metadata.append(random_selection)

        #     time_step_idz = 0
        #     for timestep_idx, (tileVH, tileVV) in enumerate(random_selection):
        #         vh, vv = self.__load_vh_vv(tileVH, tileVV)
        #         tile_array = self.__create_sample_timestep(vh, vv)

        #         if self.clip_range:
        #             min_, max_ = self.clip_range
        #             np.clip(X, min_, max_, out=X)

        #         time_series_stack[:,:,timestep_idx*2:timestep_idx*2+2] = tile_array
        #         time_step_idy += 2

        #     # if we end up with a stack with less than the set amount of timesteps, 
        #     # append existing elements until we get enough timesteps

        #     # We have three options to work around this problem of variable timesteps we either
        #         # Removing rows with missing values.
        #         # Mark and learn missing values.
        #         # Mask and learn without missing values.
        #     if time_step_idy < self.time_steps*self.n_channels:
        #         # idx = len(time_series_stack)
        #         # pad out the sequence with the last time step if there aren't enough timesteps
        #         temp = time_series_stack[:,:,-2:]
        #         while(time_step_idy != self.time_steps*self.n_channels):
        #             time_series_stack[:,:,time_step_idy:time_step_idy+2] = temp
        #             time_step_idy += 2

        #     #convert list of vv vh composites to numpy array
        #     x_stack = np.stack(time_series_stack, axis=0)

        #     mask_array = self.__get_mask(sample_subset_prefix, frame_number)
        #     # one_hot = self.__to_one_hot(mask_array, self.n_classes)

        #     # Augment data
        #     if self.training:
        #         x_stack, mask_array = self.__augment_training_data(x_stack, mask_array)

            # X[sample_idx,] = x_stack
            # y[sample_idx,] = mask_array.reshape((*self.output_dim, self.output_channels))

        #keep track of testing data
        if not self.training:
            self.__setBatchMetadata(sample_metadata)

        return np.nan_to_num(X, nan=0, copy=False), np.nan_to_num(y, nan=0, copy=False)


    # Non-keras.utils.Sequence functions

    # Augment training data using albumentations library
    def __augment_training_data(self, sample_stack, mask):
        augmentation_input = {}

        aug_output = self.augment(image=sample_stack[:,:,:sample_stack.shape[-1]], mask=mask)

        # non-channel packing implementation, for 4D input (timesteps, widht, height, channels)
        # for img_idx in range(sample_stack.shape[-1]):
        #     augmentation_input[f"image{img_idx}"] = sample_stack[:,:, idx:idx+self.n_channels]
        # x_stack_augmented = np.stack([aug_output[f"image{img_idx}"] for img_idx in range(len(sample_stack))])

        image_augmented = aug_output["image"]
        mask_augmented = aug_output["mask"]

        return image_augmented, mask_augmented

    # loads vh and vv tifs from dataset relative filepath (ie: test/WA_2018/S1A_VH_ulx_0_uly_0.tif)
    def __load_vh_vv(self, vh_dataset_path:str, vv_dataset_path: str):
        try:
            with gdal_open(os.path.join(self.dataset_directory,vh_dataset_path)) as f:
                vh = f.ReadAsArray()
        except FileNotFoundError:
            print(f"Missing file {os.path.join(self.dataset_directory,vh_dataset_path)}")
            # continue
        try:
            with gdal_open(os.path.join(self.dataset_directory,vv_dataset_path)) as f:
                vv = f.ReadAsArray()
        except FileNotFoundError:
            print(f"Missing file {os.path.join(self.dataset_directory,vv_dataset_path)}")
            # continue

        return vh, vv
    
    # Takes vh and vv arrays and either stacks them if input_channels = 2, or uses the one with a wider value range if input_channels = 1
    def __create_sample_timestep(self, vh, vv):
        tile_array = []
        if self.n_channels == 2:
            tile_array = np.stack((vh, vv), axis=2).astype('float32')
        else:
            vh = np.array(vh).astype('float32').reshape(*self.dim, self.n_channels)
            vv = np.array(vv).astype('float32').reshape(*self.dim, self.n_channels)

            if np.ptp(vh) > np.ptp(vv):
                tile_array = vh
            else:
                tile_array = vv

            # tile_array = np.stack((vh, vv), axis=2).astype('float32')
        # if np.ptp(tile_array) == 0:
        #     tile_array = np.ones(shape=(*self.dim, self.n_channels)).astype('float32')
        # else:
        #     tile_array = (tile_array - np.min(tile_array))/ np.ptp(tile_array)
        
        return tile_array
    
    # given the prefix and frame index
    def __get_mask(self, sample_subset_prefix: str, frame_number):
        # get corresponding mask prefix (ie: CA_2019, and grab the corresponding mask file)
        subset_name = f"{'_'.join(sample_subset_prefix.split('_')[:-1])}"
        subset_mask_dir_name = f"{subset_name}_masks"
        file_name = f"CDL_{subset_name}_mask_{frame_number}.tif"
        target_folder = "train"

        if not self.training:
            target_folder = "test"

        mask_path = os.path.join(self.dataset_directory, target_folder, subset_mask_dir_name, file_name)

        mask = np.zeros((*self.output_dim, self.output_channels)).astype('float32')

        try:
            with gdal_open(mask_path) as f:
                mask = np.array(f.ReadAsArray()).astype('float32')    
        except FileNotFoundError:
            print(f"Mask {mask_path} missing")

        return mask
        

    # Encodes the time series mask, an image array of shape (dim, dim, 1), into a one-hot encoding form for Categorical CrossEntropy loss with softmax activation.
    # Each unique pixel value represents a category, and the amount of unique pixel values should = the number of categories, including the background
    # For each unique category we create a channel, and for each pixel in that category we assign a 1 to that pixel position in it's corresponding channel.
    def __to_one_hot(self, mask_array, n_classes):
        one_hot = []
        if n_classes > 2:
            one_hot = np.zeros((mask_array.shape[0], mask_array.shape[1], n_classes))
            for i, unique_value in enumerate(np.unique(mask_array)):
                one_hot[:, :, i][mask_array == unique_value] = 1
        else:
            one_hot = mask_array.reshape(*self.output_dim, 1) #[:, :, 0].reshape(*self.output_dim, 1)
        return one_hot

    def __random_frame_sample(self, vh_vv_pairs: List):
        random_selection = []
        if self.time_steps == 1:
            return [vh_vv_pairs[-1]]
            # return random.sample(vh_vv_pairs, 1)
        if len(vh_vv_pairs) >= 6 and self.time_steps > 2 and self.time_steps < len(vh_vv_pairs):
            # to ensure we get a good sample of a beginning middle and end
            one_third = len(vh_vv_pairs)//3
            beginning = vh_vv_pairs[:one_third]
            middle = vh_vv_pairs[one_third:2*one_third]
            end = vh_vv_pairs[2*one_third:]

            # favor picking more time steps from the middle of the stack if timesteps can't be evenly distributed across
            one_third_time_steps = self.time_steps // 3
            middle_third_time_steps = self.time_steps - 2 * one_third_time_steps

            random_selection.extend(random.sample(beginning, one_third_time_steps))
            random_selection.extend(random.sample(middle, min(middle_third_time_steps, len(middle))))
            random_selection.extend(random.sample(end, one_third_time_steps))
        elif len(vh_vv_pairs) >= 3 and self.time_steps < len(vh_vv_pairs):
            # if we have a short enough sample, just use first, some arbritary selection of the middle, and the final frame
            random_selection = [vh_vv_pairs[0]]
            middle = vh_vv_pairs[1:-1]
            random_selection.extend(random.sample(middle, self.time_steps - 2))
            random_selection.append(vh_vv_pairs[-1])
        else:
            random_selection = random.sample(vh_vv_pairs, min(self.time_steps, len(vh_vv_pairs)))


        return random_selection

    # accessor to get metadata, ordered by input, contains strings of file paths and their 
    # crop mask in the order they're fed to the model
    def getBatchMetadata(self) -> List[Tuple[List[Tuple[str, str]], str]]:
        return self.metadata

    # when each batch begins we record what files are passed to the model
    def __setBatchMetadata(self, batch: Tuple[List[Tuple[str, str]], str]):
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
