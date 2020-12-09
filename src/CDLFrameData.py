from typing import Dict, List, Optional, Tuple
import os
import random
import numpy as np
from src.gdal_wrapper import gdal_open
from src.config import NETWORK_DEMS, N_CHANNELS
import cv2
from albumentations import (
    Compose, ToFloat
)


class CDLFrameData():
    def __init__(self, sub_dataset: str, frame_index_key: str, dataset_directory):
        self.sub_dataset = sub_dataset
        self.frame_index_key = frame_index_key
        self.file_paths = []
        self.dataset_directory = dataset_directory
    
    def get_sub_dataset_name(self) -> str:
        return self.sub_dataset

    def get_frame_index_key(self) -> str:
        return self.frame_index_key

    def load_timeseries_frame_data(self, data_paths, dims, n_channels, timesteps, augmentations=Compose([ToFloat(max_value=255, p=0.0),
        ])):

        # randomly choose time steps to include in timeseries sample
        random_selection = self.__random_frame_sample(data_paths, timesteps)

        # if the timeseries has fewer timesteps than what we want to train the model with, we pad it
        # with existing randomly selected timesteps
        if len(random_selection) < timesteps:
            random_selection = self.__extend_sample_timesteps(random_selection, timesteps)

        # train model on chronologically ordered data
        random_selection = self.__sort_data(random_selection)
        # load the actual data from our randomly chosen files
        timeseries_sample = self.__create_timeseries_sample(random_selection, dims, n_channels, timesteps)

        # keep track of file paths for model test file metadata
        self.file_paths = random_selection

        mask_array = self.__get_mask(self.sub_dataset, self.frame_index_key)

        # convert mask to one_hot encoding for categorical data (multi-class classification, non-binary)
        # mask_array = self.__to_one_hot(mask_array, n_classes, dims)

        x_stack, mask_array = self.__augment_data(timeseries_sample, mask_array, augmentations)

        return x_stack, mask_array.reshape((*dims, 1))

    def get_metadata_paths(self):
        return self.file_paths

    def __random_frame_sample(self, vh_vv_pairs: List, time_steps: int):
        random_selection = []
        if time_steps == 1:
            return [vh_vv_pairs[-1]]
            # return random.sample(vh_vv_pairs, 1)
        if len(vh_vv_pairs) >= 6 and time_steps > 2 and time_steps < len(vh_vv_pairs):
            # to ensure we get a good sample of a beginning middle and end
            one_third = len(vh_vv_pairs)//3
            beginning = vh_vv_pairs[:one_third]
            middle = vh_vv_pairs[one_third:2*one_third]
            end = vh_vv_pairs[2*one_third:]

            # favor picking more time steps from the middle of the stack if timesteps can't be evenly distributed across
            one_third_time_steps = time_steps // 3
            middle_third_time_steps = time_steps - 2 * one_third_time_steps

            random_selection.extend(random.sample(beginning, one_third_time_steps))
            random_selection.extend(random.sample(middle, min(middle_third_time_steps, len(middle))))
            random_selection.extend(random.sample(end, one_third_time_steps))
        elif len(vh_vv_pairs) >= 3 and time_steps < len(vh_vv_pairs):
            # if we have a short enough sample, just use first, some arbritary selection of the middle, and the final frame
            random_selection = [vh_vv_pairs[0]]
            middle = vh_vv_pairs[1:-1]
            random_selection.extend(random.sample(middle, time_steps - 2))
            random_selection.append(vh_vv_pairs[-1])
        else:
            random_selection = random.sample(vh_vv_pairs, min(time_steps, len(vh_vv_pairs)))

        return random_selection
   
   #pad out data with random timestep sampling in case sample doesn't has less than user defined time steps
    def __extend_sample_timesteps(self, vh_vv_pairs: List, time_steps: int):
        output = []
        output.extend(vh_vv_pairs)

        while len(output) < time_steps:
            output.extend(random.sample(vh_vv_pairs, min(time_steps - len(vh_vv_pairs), len(vh_vv_pairs))))  
        
        return output[:time_steps]

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
    def __create_sample_timestep(self, vh, vv, n_channels, timestep_dims = (NETWORK_DEMS, NETWORK_DEMS, N_CHANNELS)):
        tile_array = []
        if n_channels == 2:
            tile_array = np.stack((vh, vv), axis=2).astype('float32')
        else:
            vh = np.array(vh).astype('float32').reshape((*timestep_dims[:-1], 1))
            vv = np.array(vv).astype('float32').reshape((*timestep_dims[:-1], 1))

            if np.ptp(vh) > np.ptp(vv):
                tile_array = vh
            else:
                tile_array = vv
        
        return tile_array

    # given the prefix and frame index, find the corresponding mask over the given area
    def __get_mask(self, sample_subset_prefix: str, frame_number, mask_shape=(NETWORK_DEMS, NETWORK_DEMS, 1)):
        # get corresponding mask prefix (ie: CA_2019, and grab the corresponding mask file)
        subset_name = f"{'_'.join(sample_subset_prefix.split('_')[:-1])}"
        subset_mask_dir_name = f"{subset_name}_masks"
        file_name = f"CDL_{subset_name}_mask_{frame_number}.tif"
        target_folder = "masks"

        mask_path = os.path.join(self.dataset_directory, target_folder, subset_mask_dir_name, file_name)

        mask = np.zeros(mask_shape).astype('float32')

        try:
            with gdal_open(mask_path) as f:
                mask = np.array(f.ReadAsArray()).astype('float32')    
        except FileNotFoundError:
            print(f"Mask {mask_path} missing")

        return mask

    # Augment training data using albumentations library
    def __augment_data(self, sample_stack, mask, augmentation):
        augmentation_input = {}

        aug_output = augmentation(image=sample_stack[:,:,:sample_stack.shape[-1]], mask=mask)

        # non-channel packing implementation, for 4D input (timesteps, widht, height, channels)
        # for img_idx in range(sample_stack.shape[-1]):
        #     augmentation_input[f"image{img_idx}"] = sample_stack[:,:, idx:idx+self.n_channels]
        # x_stack_augmented = np.stack([aug_output[f"image{img_idx}"] for img_idx in range(len(sample_stack))])

        image_augmented = aug_output["image"]
        mask_augmented = aug_output["mask"]

        return image_augmented, mask_augmented

    def __sort_data(self, time_step_paths):
        # Ignore S1B / S1A prefix in sorting
        time_step_paths.sort(key=lambda pair: "_".join(pair[0].split("_")[1:]))
        return time_step_paths

    def __create_timeseries_sample(self, timeseries_sample_prefix_and_frames, dims, n_channels, timesteps):
        timeseries_sample = np.zeros((*dims, n_channels*timesteps), dtype=np.float32)

        for timestep_idx, (tileVH, tileVV) in enumerate(timeseries_sample_prefix_and_frames):
            vh, vv = self.__load_vh_vv(tileVH, tileVV)
            tile_array = self.__create_sample_timestep(vh, vv, n_channels)

            # if self.clip_range:
            #     min_, max_ = self.clip_range
            #     np.clip(X, min_, max_, out=X)

            timeseries_sample[:,:,timestep_idx*2:timestep_idx*2+2] = tile_array
        
        return timeseries_sample

    # Encodes the time series mask, an image array of shape (dim, dim, 1), into a one-hot encoding form for Categorical CrossEntropy loss with softmax activation.
    # Each unique pixel value represents a category, and the amount of unique pixel values should = the number of categories, including the background
    # For each unique category we create a channel, and for each pixel in that category we assign a 1 to that pixel position in it's corresponding channel.
    def __to_one_hot(self, mask_array, n_classes, output_dim):
        one_hot = []
        if n_classes > 2:
            one_hot = np.zeros((mask_array.shape[0], mask_array.shape[1], n_classes))
            for i, unique_value in enumerate(np.unique(mask_array)):
                one_hot[:, :, i][mask_array == unique_value] = 1
        else:
            one_hot = mask_array.reshape(output_dim, 1) #[:, :, 0].reshape(*self.output_dim, 1)
        return one_hot