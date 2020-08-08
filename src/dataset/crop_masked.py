"""
crop_masked.py contains the code for preparing a Time Distributed masked data set, and loading
the prepared data set for use.
"""

import os
import tensorflow as tf
from math import floor
import re
from math import floor
from typing import Generator, Optional, Tuple

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, Iterator
from keras.preprocessing.sequence import TimeseriesGenerator
from osgeo import gdal

from ..SARTimeseriesGenerator import SARTimeseriesGenerator
from ..gdal_wrapper import gdal_open
from ..asf_typing import MaskedDatasetMetadata, MaskedTimeseriesMetadata
from .common import dataset_dir, valid_image
from ..config import NETWORK_DEMS, TIMESTEPS

TILE_REGEX = re.compile(r"(.*)\.vh(.*)\.(tiff|tif|TIFF|TIF)")
TITLE_TIME_SERIES_REGEX = re.compile(r"(.*)\_VH(.*)\.(tiff|tif|TIFF|TIF)")
# TITLE_TIME_SERIES_REGEX = re.compile(r"(.*)\_VH\.(tiff|tif|TIFF|TIF)")


def load_timeseries_dataset(dataset: str) -> Tuple[Iterator]:

    train_metadata, test_metadata = make_timeseries_metadata(dataset)

    sample_size = len(train_metadata[0])
    time_steps = len(train_metadata[0][0][0])
    print("Sample Size:\t", sample_size)
    print("Time Steps:\t", time_steps)

    # pre-allocate for inserting by index
    # x_train = np.empty((sample_size, time_steps, NETWORK_DEMS, NETWORK_DEMS, 2))
    # y_train = np.empty((sample_size, 1, NETWORK_DEMS, NETWORK_DEMS, 1))

    # for idx, (time_stack, mask) in enumerate(generate_timeseries_from_metadata(train_metadata, clip_range=(0, 2))):
    #     x_train[idx, :] = time_stack
    #     y_train[idx, :] = mask

    # print("\nX data shape:\t", x_train.shape)
    # print("Y data shape:\t", y_train.shape)
    validation_split = .25
    split_index = floor(sample_size * validation_split)
    train_iter = SARTimeseriesGenerator(
        train_metadata[0][:-split_index],
        batch_size=1,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=2,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        n_classes=2,
        shuffle=False)
    validation_iter = SARTimeseriesGenerator(
        train_metadata[0][-split_index:],
        batch_size=1,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=2,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        n_classes=2,
        shuffle=False)
    return train_iter, validation_iter

def load_test_timeseries_dataset(dataset: str):
    train_metadata, test_metadata = make_timeseries_metadata(dataset)

    sample_size = len(train_metadata[0])
    time_steps = len(train_metadata[0][0][0])
    print("Sample Size:\t", sample_size)
    print("Time Steps:\t", time_steps)
    
    test_iter = SARTimeseriesGenerator(
        test_metadata[0],
        batch_size=1,
        dim=(NETWORK_DEMS, NETWORK_DEMS),
        time_steps=time_steps,
        n_channels=2,
        output_dim=(NETWORK_DEMS, NETWORK_DEMS),
        output_channels=1,
        n_classes=2,
        shuffle=False)
    
    return test_metadata, test_iter


def load_replace_timeseries_data(
    dataset: str,
    dems=NETWORK_DEMS
) -> Tuple[Iterator, MaskedDatasetMetadata]:

    #replace_gen = ImageDataGenerator(rescale=10)
    metadata, _ = make_timeseries_metadata(dataset, edit=True)

    # Load the entire dataset into memory
    # batch_size = len(metadata[0])
    batch_size = 1
    time_steps = len(metadata[0][0][0])
    print("Batch Size:\t", batch_size)
    print("Time Steps:\t", time_steps)
    # x_train = np.empty((1862, 786432))
    x_train = np.empty((266 * time_steps, NETWORK_DEMS, NETWORK_DEMS, 2))
    # x_train = np.empty((266, 9, 512, 512, 3))
    # y_train = []
    # y_train = np.empty((1862, 262144))
    y_train = np.empty((266 * time_steps, NETWORK_DEMS, NETWORK_DEMS, 1))
    # y_train = np.empty((266, 9, 512, 512, 1))
    for idx, (time_stack, mask) in enumerate(generate_timeseries_from_metadata(metadata, clip_range=(0, 2))):
        # x_train.concat(time_stack)
        # y_train.concat(mask)
        x_train[idx, :] = time_stack
        y_train[idx, :] = mask
    # replace_iter = TimeseriesGenerator(x_train, batch_size=1, length = time_steps)
    replace_iter = TimeseriesGenerator(
        x_train, batch_size=batch_size, stride=time_steps, sampling_rate=1, targets=y_train, length=time_steps)
    # replace_iter = replace_gen.flow(
    #    np.array(x_replace), y=np.array(y_replace), batch_size=1, shuffle=False
    # )

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

    # expectation that train and test will be root dataset directories
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
                                if re.match(fr"(.*)\_VV{end}\.(tif|tiff)", tileVV) and validate_image(timeseries_path, file_dir, tileVV)
                            ]
                            frames.append(end)
                        else:
                            continue

                        # for testing purposes
                        mask_name = f"CDL_IA_2019_mask{end}.{ext}"

                        
                        # Get VV VH Pair
                        if(len(VV_Tiles) > 0):
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

                        # get mask name for specific frame
                        for mask in sorted(files):
                            if re.search(fr"(.*)\_mask{end}\.(tif|tiff)", mask):
                                mask_name = mask
                                # print(mask_name)
                                break
                        # The timestack with mask tuple(list(tuple(vv, vh)), mask)
                        data_frame = (
                            tile_time_series_data,
                            os.path.join(
                                timeseries_path, file_dir, mask_name
                            )
                        )
                        print("LENGTH OF DATA:\t", len(data_frame[0]))
                        # print("LENGTH OF DATA:\t", len(data[0][0]), "\n")
                        if len(data_frame[0]) != 0:
                            data.append(data_frame)


                    if edit:
                        if data_dir == 'test' or data_dir == 'train' and len(data[0]) !=0:
                            train_metadata.append(data)
                    else:
                        if data_dir == 'train' and len(data[0]) !=0:
                            train_metadata.append(data)
                        elif data_dir == 'test' and len(data[0]) !=0:
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

                y = np.array(mask_array).astype('float32').reshape(512, 512, 1)

                y_stack = []
                # for zed in range(x_stack.shape[0]):
                #     y_stack.append(y.reshape(512, 512, 1))

                y_stack = np.array(y_stack)
                # y_stack = np.array(y_stack).reshape(512, 512, 1)
                # print(x_stack.shape, "\t", y.shape)
                yield (x_stack, y)
                # for zed in range(len(x_stack)):
                #     yield(x_stack[zed], y_stack[zed].reshape(512, 512, 1))
                # yield (x_stack, y_stack)

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