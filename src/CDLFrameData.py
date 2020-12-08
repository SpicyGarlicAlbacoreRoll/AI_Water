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
    
    def get_dataset_name(self) -> str:
        return self.sub_dataset

    def get_frame_index_key(self) -> str:
        return self.frame_index_key

    def load_timeseries_frame_data(self, data_paths, dims, n_channels, timesteps, augmentations=Compose([ToFloat(max_value=255, p=0.0),
        ])):
        time_series_stack = np.zeros((*dims, n_channels*timesteps), dtype=np.float32)
        time_step_idy = 0
        # print(data_paths)
        random_selection = self.__random_frame_sample(data_paths, timesteps)

        # Ignore S1B / S1A prefix in sorting
        random_selection.sort(key=lambda pair: pair[0].split("_")[1:])

        self.file_paths = random_selection

        time_step_idz = 0
        # print(random_selection)
        for timestep_idx, (tileVH, tileVV) in enumerate(random_selection):
            vh, vv = self.__load_vh_vv(tileVH, tileVV)
            tile_array = self.__create_sample_timestep(vh, vv, n_channels)

            # if self.clip_range:
            #     min_, max_ = self.clip_range
            #     np.clip(X, min_, max_, out=X)

            time_series_stack[:,:,timestep_idx*2:timestep_idx*2+2] = tile_array
            time_step_idy += 2

        # if we end up with a stack with less than the set amount of timesteps, 
        # append existing elements until we get enough timesteps

        # We have three options to work around this problem of variable timesteps we either
            # Removing rows with missing values.
            # Mark and learn missing values.
            # Mask and learn without missing values.
        if time_step_idy < timesteps*n_channels:
            # idx = len(time_series_stack)
            # pad out the sequence with the last time step if there aren't enough timesteps
            temp = time_series_stack[:,:,-2:]
            while(time_step_idy != timesteps*n_channels):
                time_series_stack[:,:,time_step_idy:time_step_idy+2] = temp
                time_step_idy += 2

        #convert list of vv vh composites to numpy array
        x_stack = np.stack(time_series_stack, axis=0)

        mask_array = self.__get_mask(self.sub_dataset, self.frame_index_key)
        # one_hot = self.__to_one_hot(mask_array, self.n_classes)

        x_stack, mask_array = self.__augment_data(x_stack, mask_array, augmentations)

        return x_stack, mask_array.reshape((NETWORK_DEMS, NETWORK_DEMS, 1))

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

            # tile_array = np.stack((vh, vv), axis=2).astype('float32')
        # if np.ptp(tile_array) == 0:
        #     tile_array = np.ones(shape=(*self.dim, self.n_channels)).astype('float32')
        # else:
        #     tile_array = (tile_array - np.min(tile_array))/ np.ptp(tile_array)
        
        return tile_array

    # given the prefix and frame index
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