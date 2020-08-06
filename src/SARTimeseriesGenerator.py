# Basic class organization based on Shervine Amidi's implementation,
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
from src.gdal_wrapper import gdal_open
from typing import Optional, Tuple
import keras

class SARTimeseriesGenerator(keras.utils.Sequence):
    def __init__(self, time_series_mask_list, batch_size=32, dim=(512, 512), time_steps=1, n_channels=2, output_dim=(512, 512), output_channels=1, n_classes=2, shuffle=True, clip_range: Optional[Tuple[float, float]] = None):
        self.list_IDs = time_series_mask_list
        # self.masks = masks

        self.batch_size = batch_size
        self.dim = dim
        self.time_steps = time_steps
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.clip_range = clip_range
        self.on_epoch_end()

    def __len__(self):
        #Returns amount of batches per epochs
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # get indices in current batch
        indexes = self.indexes[index*self.batch_size : (index+1) * self.batch_size]

        # get list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # (samples, timesteps, width, height, channels)
        X = np.empty((self.batch_size, self.time_steps, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, 1, *self.output_dim, self.output_channels), dtype=np.float32)

        for sample_idx, (time_series, mask) in enumerate(list_IDs_temp):
            time_series_stack = []
            time_series_mask = []
            
            for tileVH, tileVV in sorted(time_series):
                try:
                    with gdal_open(tileVH) as f:
                        vh = f.ReadAsArray()
                except FileNotFoundError:
                    continue
                try:
                    with gdal_open(tileVV) as f:
                        vv = f.ReadAsArray()
                except FileNotFoundError:
                    continue
                
                tile_array = np.stack((vh, vv), axis=2).astype('float32')

                if self.clip_range:
                    min_, max_ = self.clip_range
                    np.clip(x, min_, max_, out=x)
                
                time_series_stack.append(tile_array)
            
            if len(time_series_stack) != 0:
                #convert list of vv vh composites to numpy array
                x_stack = np.stack(time_series_stack, axis=0).astype('float32')

                with gdal_open(mask) as f:
                    mask = f.ReadAsArray()
                
                mask_array = np.array(mask).astype('float32').reshape(512, 512, 1)

                X[sample_idx,] = x_stack
                y[sample_idx,] = mask_array
        
        return X, y