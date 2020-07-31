"""
crop_masked.py contains the code for preparing a Time Distributed masked data set, and loading
the prepared data set for use.
"""

import os
import re
from typing import Generator, Optional, Tuple

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.preprocessing.sequence import TimeseriesGenerator
from osgeo import gdal

from ..gdal_wrapper import gdal_open
from ..asf_typing import MaskedDatasetMetadata, MaskedTimeseriesMetadata
from .common import dataset_dir, valid_image
from ..config import NETWORK_DEMS, TIMESTEPS

TILE_REGEX = re.compile(r"(.*)\.vh(.*)\.(tiff|tif|TIFF|TIF)")
TITLE_TIME_SERIES_REGEX = re.compile(r"(.*)\_VH(.*)\.(tiff|tif|TIFF|TIF)")
# TITLE_TIME_SERIES_REGEX = re.compile(r"(.*)\_VH\.(tiff|tif|TIFF|TIF)")


def load_timeseries_dataset(dataset: str) -> Tuple[Iterator, Iterator]:
    train_gen = ImageDataGenerator(rescale=10)
    test_gen = ImageDataGenerator(rescale=10)

    train_metadata, test_metadata = make_timeseries_metadata(dataset)
    # Load the entire dataset into memory
    # x_train = []
    # x_train = np.array(x_train).reshape(1, 512, 512, 2)
    batch_size = len(train_metadata[0])
    batch_size = 1
    time_steps = len(train_metadata[0][0][0])
    print("Batch Size:\t", batch_size)
    print("Time Steps:\t", time_steps)
    # x_train = np.empty((1862, 786432))
    x_train = np.empty((266 * time_steps, NETWORK_DEMS, NETWORK_DEMS, 2))
    # x_train = np.empty((266, 9, 512, 512, 3))
    # y_train = []
    # y_train = np.empty((1862, 262144))
    y_train = np.empty((266 * time_steps, NETWORK_DEMS, NETWORK_DEMS, 1))
    # y_train = np.empty((266, 9, 512, 512, 1))
    for idx, (time_stack, mask) in enumerate(generate_timeseries_from_metadata(train_metadata, clip_range=(0, 2))):
        # x_train.concat(time_stack)
        # y_train.concat(mask)
        x_train[idx, :] = time_stack
        y_train[idx, :] = mask

    # x_test = np.empty((2660, 512, 512, 3))
    # # x_test = np.empty((266, 10, 512, 512, 3))
    # # y_test = []
    # y_test = np.empty((2660, 512, 512, 1))
    # # y_test = np.empty((266, 10, 512, 512, 1))

    # for time_stack, mask in generate_timeseries_from_metadata(test_metadata, clip_range=(0, 2)):
    #     x_test[idx, :] = time_stack
    #     y_test[idx, :] = mask


    del train_metadata[:]
    del train_metadata
    del test_metadata[:]
    del test_metadata

    # Needed to work with Time Distributed Layers
    # https://stackoverflow.com/questions/58948739/reshaping-images-for-input-into-keras-timedistributed-function
    # train_gen = TimeseriesGenerator()
    # test_gen = TimeseriesGenerator()
    # print(x_test[0]8.shape)
    # train_target_length = x_train.shape[0]
    # test_target_length = x_test.shape[0]

    # x_train = np.stack(np.array(x_train))
    # x_test = np.stack(np.array(x_test))
    # y_train = np.stack(np.array(y_train))
    # y_test = np.stack(np.array(y_test))
    print("Generation successful")
    # x_train = np.array(x_train).reshape(4788, 512, 512, 2)
    # x_test = np.array(x_test).reshape(4788, 512, 512, 2)
    print("x_train shape:\t", x_train.shape)
    # print("x_test shape:\t", x_test.shape)
    # zed = input()
    # y_train = np.array(y_train).reshape(4788, 512, 512, 2)
    # y_test = np.array(y_test).reshape(4788, 512, 512, 2)
    print("y_train:\t", y_train.shape)
    # print("y_test:\t", y_test.shape)
    # zed=input()
    # x_train = np.array(x_train)[0]
    # x_test = np.array(x_test)[0]
    # y_train = np.expand_dims(np.array(y_train)[0], axis=0)
    # y_test = np.expand_dims(np.array(y_test)[0], axis=0)
    # y_train = np.stack((y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train, y_train), axis=0)
    # y_test = np.stack((y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test, y_test), axis = 0)
    
    # print("X shape:\t", x_train.shape)
    # print("Y shape:\t", y_train.shape)
    # for idx, t in enumerate(x_train):
    #     print(t.shape)
    #     train_gen = TimeseriesGenerator(t, batch_size=1, targets=np.stack((y_train, y_train)), length = 1)

    # train_gen = TimeseriesGenerator(x_train, batch_size=1, targets=y_train, length = 1)
    # print("train_gen successful!")
    # # for idx, t in enumerate(x_test):
    # #     print(t.shape)
    # #     test_gen = TimeseriesGenerator(t, batch_size=1, targets=np.stack((y_test, y_test, y_test)), length = 1)
    # test_gen = TimeseriesGenerator(x_test, batch_size=1, targets=y_test, length = 2)
    # print("test_gen successful!")
    # train_gen.fit(x_train)
    zedX = np.array(x_train)[0]
    zedY = np.array(y_train)[0]
    print(zedX.shape)
    print(zedY.shape)
    # zedY = np.expand_dims(np.array(y_train)[0], axis=0)
    
    
    # train_iter = train_gen.flow(
    #     x=zedX, y=zedY, batch_size=1
    # )
    
    # train_iter = TimeseriesGenerator(x_train, batch_size=time_steps, targets=y_train, length = 1)

    #stride: the stride between samples (in this case, the amount of timesteps in a sample)
    #length: is the length of output sequences
    train_iter = TimeseriesGenerator(x_train, batch_size=batch_size, stride=time_steps, sampling_rate=1, targets=y_train, length = time_steps)
    print(len(train_iter))

    # test_gen.fit(train_iter)

    # print("train_gen successful!")
    # test_iter = test_gen.flow(
    #     x=x_test, y=y_test, batch_size=1, shuffle = False
    # )
    print("skipping test_gen...")

    return train_iter, train_iter
    # return test_iter, train_iter


def load_replace_timeseries_data(
    dataset: str,
    dems=NETWORK_DEMS
) -> Tuple[Iterator, MaskedDatasetMetadata]:

    #replace_gen = ImageDataGenerator(rescale=10)
    metadata, _ = make_timeseries_metadata(dataset, edit=True)

    # Load the entire dataset into memory
    batch_size = len(metadata[0])
    time_steps = len(metadata[0][0][0])
    print("Batch Size:\t", batch_size)
    print("Time Steps:\t", time_steps)
    # x_train = np.empty((1862, 786432))
    x_train = np.empty((batch_size * time_steps, NETWORK_DEMS, NETWORK_DEMS, 2))
    # x_train = np.empty((266, 9, 512, 512, 3))
    # y_train = []
    # y_train = np.empty((1862, 262144))
    y_train = np.empty((batch_size * time_steps, NETWORK_DEMS, NETWORK_DEMS, 1))
    # y_train = np.empty((266, 9, 512, 512, 1))
    for idx, (time_stack, mask) in enumerate(generate_timeseries_from_metadata(metadata, clip_range=(0, 2))):
        # x_train.concat(time_stack)
        # y_train.concat(mask)
        x_train[idx, :] = time_stack
        y_train[idx, :] = mask
    replace_iter = TimeseriesGenerator(x_train, batch_size=1, length = time_steps)

    #replace_iter = replace_gen.flow(
    #    np.array(x_replace), y=np.array(y_replace), batch_size=1, shuffle=False
    #)

    return replace_iter, metadata


def make_timeseries_metadata(
    dataset: str,
    edit: bool = False
) -> Tuple[MaskedTimeseriesMetadata, MaskedTimeseriesMetadata]:
    """ Returns two lists of metadata. One for the training data and one for the
    testing data. """
    train_metadata = []
    test_metadata = []

    dirs = ["train", "test"]

    #expectation that train and test will be root dataset directories
    for data_dir in dirs:
        for timeseries_path, timeseries_dirs, timeseries_files in os.walk(os.path.join(dataset_dir(dataset), data_dir)):
            for file_dir in timeseries_dirs:
                for data_point_path, _, files in os.walk(os.path.join(timeseries_path, file_dir)):
                    # print(files)
                    print(data_point_path)

                    # our list of time series frames + masks that will be appended to test and train metadata
                    data = []
                    
                    # keep track of frames that have already been stacked
                    frames = []
                    
                    for tile_name in files:
                        m = re.match(TITLE_TIME_SERIES_REGEX, tile_name)
                        if not m:
                            continue
                        
                        pre, end, ext = m.groups()
                        
                        tile_time_series_data = []
                        frame_index = end
                        VV_Tiles = []
                        
                        # If we've already gotten a time stack of this frame index, skip
                        if frame_index not in frames:
                            VV_Tiles = [
                                tileVV for tileVV in sorted(files)
                                if re.match(fr"(.*)\_VV{end}\.(tif|tiff)", tileVV)
                            ]
                            frames.append(end)
                        else:
                            continue

                        # for testing purposes
                        mask_name = f"CDL_IA_2019_mask{end}.{ext}"

                        # Get VV VH Pair
                        for tileVV in VV_Tiles:
                            tile_time_series_data.append(
                                (
                                    os.path.join(
                                        timeseries_path, file_dir, tileVV
                                    ),
                                    os.path.join(
                                        timeseries_path, file_dir, tileVV.replace("VV", "VH"))
                                    )
                                )
                        
                        #get mask name for specific frame
                        for mask in sorted(files):
                            if re.search(fr"(.*)\_mask{end}\.(tif|tiff)", mask):
                                mask_name = mask
                                # print(mask_name)
                                break
                        # The timestack with mask tuple(list(tuple(vv, vh)), mask)
                        data_frame = (
                                tile_time_series_data, 
                                os.path.join (
                                    timeseries_path, file_dir, mask_name
                                )
                            )

                        data.append(data_frame)

                    if edit:
                        if data_dir == 'test' or data_dir == 'train':
                            train_metadata.append(data)
                    else:
                        if data_dir == 'train':
                            print("training:\t", data[0])
                            train_metadata.append(data)
                        elif data_dir == 'test':
                            print("testing:\t", data[0])
                            test_metadata.append(data)

    return train_metadata, test_metadata


def generate_timeseries_from_metadata(
    metadata: MaskedTimeseriesMetadata,
    clip_range: Optional[Tuple[float, float]] = None,
    edit: bool = False,
    dems=NETWORK_DEMS,
    timesteps=TIMESTEPS
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """ Yield training images and masks from the given metadata. """
    output_shape = (timesteps, dems, dems, 3)
    mask_output_shape = (dems, dems, 1)
    
    for time_series_mask_pairs in metadata:
        for time_series, mask in time_series_mask_pairs:
            time_stack = []

            for tileVH, tileVV in sorted(time_series):
                tif_vh = gdal.Open(tileVH)

                comp = str(tif_vh.RasterXSize)
                if(comp != str(dems) and "mock" not in comp):
                    continue
                try:
                    with gdal_open(tileVH) as f:
                        tile_vh_array = f.ReadAsArray()
                except FileNotFoundError:
                    continue
                try:
                    with gdal_open(tileVV) as f:
                        tile_vv_array = f.ReadAsArray()
                except FileNotFoundError:
                    continue

                # blue_channel = np.multiply(np.add(tile_vh_array, tile_vv_array), 0.5)

                tile_array = np.stack((tile_vh_array, tile_vv_array), axis=2)


                # if not edit:
                #     if not valid_image(tile_array):
                #         continue
                
                x = np.array(tile_array).astype('float32')

                if clip_range:
                    min_, max_ = clip_range
                    np.clip(x, min_, max_, out=x)

                time_stack.append(x)
            
            if len(time_stack) != 0:
                x_stack = np.stack(time_stack, 0).astype('float32')
                with gdal_open(mask) as f:
                    mask_array = f.ReadAsArray()
                
                

                y = np.array(mask_array).astype('float32')

                y_stack = []
                for zed in range(x_stack.shape[0]):
                    y_stack.append(y.reshape(512, 512, 1))
                
                y_stack = np.array(y_stack)
                # y_stack = np.array(y_stack).reshape(512, 512, 1)
                # print(x_stack.shape, "\t", y.shape)
                # yield (x_stack, y_stack)
                for zed in range(len(x_stack)):
                    yield(x_stack[zed], y_stack[zed].reshape(512, 512, 1))
                # yield (x_stack, y_stack)
