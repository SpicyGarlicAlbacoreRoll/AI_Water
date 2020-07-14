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
    x_train = []
    y_train = []
    for time_stack, mask in generate_timeseries_from_metadata(train_metadata, clip_range=(0, 2)):
        x_train.append(time_stack)
        y_train.append(mask)

    x_test = []
    y_test = []
    # print(test_metadata)
    for time_stack, mask in generate_timeseries_from_metadata(test_metadata, clip_range=(0, 2)):
        x_test.append(time_stack)
        y_test.append(mask)

    # Needed to work with Time Distributed Layers
    # https://stackoverflow.com/questions/58948739/reshaping-images-for-input-into-keras-timedistributed-function
    # train_gen = TimeseriesGenerator()
    # test_gen = TimeseriesGenerator()

    # train_gen = TimeseriesGenerator(x_train,)

    train_iter = train_gen.flow(
        x=x_train, y=y_train, batch_size=1
    )
    test_iter = test_gen.flow(
        x_test, y=y_test, batch_size=1, shuffle=False
    )

    return train_iter, test_iter


def load_replace_data(
    dataset: str,
    dems=NETWORK_DEMS
) -> Tuple[Iterator, MaskedDatasetMetadata]:

    replace_gen = ImageDataGenerator(rescale=10)
    metadata, _ = make_timeseries_metadata(dataset, edit=True)

    # Load the entire dataset into memory
    x_replace = []
    y_replace = []
    for img, mask in generate_timeseries_from_metadata(
        metadata,
        edit=True,
        clip_range=(0, 2),
        dems=dems
    ):
        x_replace.append(img)
        y_replace.append(mask)

    replace_iter = replace_gen.flow(
        np.array(x_replace), y=np.array(y_replace), batch_size=1, shuffle=False
    )

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
                        mask_name = f"{pre}_VH{end}.{ext}"

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
    output_shape = (timesteps, dems, dems, 2)
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

                tile_array = np.stack((tile_vh_array, tile_vv_array), axis = 2)

                # if not edit:
                #     if not valid_image(tile_array):
                #         continue
                
                x = np.array(tile_array).astype('float32')

                if clip_range:
                    min_, max_ = clip_range
                    np.clip(x, min_, max_, out=x)

                time_stack.append(x)
            
            if len(time_stack) != 0:
                x_stack = np.stack(time_stack, axis=0)
                with gdal_open(mask) as f:
                    mask_array = f.ReadAsArray()
            
                y = np.array(mask_array).astype('float32')
                yield (x_stack, y.reshape(mask_output_shape))