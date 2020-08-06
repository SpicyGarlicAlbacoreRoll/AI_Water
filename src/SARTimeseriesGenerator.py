# Basic class organization based on Shervine Amidi's implementation,
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import keras

class SARTimeseriesGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, masks, batch_size=32, dim=(512, 512), time_steps=1, n_channels=2, output_dim=(512, 512), output_channels=1, n_classes=2, shuffle=True):
        self.list_IDs = list_IDs
        self.masks = masks
        self.batch_size = batch_size
        self.dim = dim
        self.time_steps = time_steps
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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

        for idx, id in list_IDs_temp:
            temp = 0
        
        return X, y