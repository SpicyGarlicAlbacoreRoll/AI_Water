"""
crop_masked.py contains the code for preparing a Time Distributed masked data set, and loading
the prepared data set for use.
"""

import os
import re
from typing import Generator, Optional, Tuple

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, Iterator
from osgeo import gdal

from ..gdal_wrapper import gdal_open
from ..asf_typing import MaskedDatasetMetadata, MaskedTimeseriesMetadata
from .common import dataset_dir, valid_image
from ..config import NETWORK_DEMS, TIMESTEPS

TILE_REGEX = re.compile(r"(.*)\.vh(.*)\.(tiff|tif|TIFF|TIF)")
TITLE_TIME_SERIES_REGEX = re.compile(r"(.*)\_VH\.(tiff|tif|TIFF|TIF)")

def load_timeseries_dataset(dataset: str) -> Tuple[Iterator, Iterator]:
    train_gen = ImageDataGenerator(rescale=10)
    test_gen = ImageDataGenerator(rescale=10)

    train_metadata, test_metadata = make_timerseries_metadata(dataset)
    # Load the entire dataset into memory
    x_train = []
    y_train = []
    print("\nTrain metadata mask")
    print(train_metadata)
    print("\nTest metadata mask")
    print(test_metadata)
    for img, mask in generate_timeseries_from_metadata(train_metadata, clip_range=(0, 2)):
        x_train.append(img)
        y_train.append(mask)

    x_test = []
    y_test = []

    for img, mask in generate_timeseries_from_metadata(test_metadata, clip_range=(0, 2)):
        x_test.append(img)
        y_test.append(mask)

    train_iter = train_gen.flow(
        np.array(x_train), y=np.array(y_train), batch_size=16
    )
    test_iter = test_gen.flow(
        np.array(x_test), y=np.array(y_test), batch_size=1, shuffle=False
    )

    return train_iter, test_iter


def load_replace_data(
    dataset: str,
    dems=NETWORK_DEMS
) -> Tuple[Iterator, MaskedDatasetMetadata]:

    replace_gen = ImageDataGenerator(rescale=10)
    metadata, _ = make_timerseries_metadata(dataset, edit=True)

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


def make_timerseries_metadata(
    dataset: str,
    edit: bool = False
) -> Tuple[MaskedTimeseriesMetadata, MaskedTimeseriesMetadata]:
    """ Returns two lists of metadata. One for the training data and one for the
    testing data. """
    train_metadata = []
    test_metadata = []

    for dirpath, dirnames, filenames in os.walk(dataset_dir(dataset)): #for folders in new_test_set
        for data_dir in dirnames:   #for folders in dataset
            if data_dir == 'test' or data_dir == 'train':
                for idx, (timeseries_path, _, timeseries_filenames) in enumerate(os.walk(os.path.join(dirpath, data_dir))):
                    print("FFFFFFFFFFFF", _)
                    time_series_data = []
                    timeseries_mask = ""
                    crop_mask_path = ""
                    for name in sorted(timeseries_filenames):
                        # print(name)
                        crop_mask_file = re.search("mask.tif", name)

                        if crop_mask_file:
                            crop_mask_path = os.path.join(dirpath, timeseries_path, crop_mask_file.group())

                        # skip to avoid double composite pair
                        m = re.match(TITLE_TIME_SERIES_REGEX, name)
                        if not m:
                            continue
                        

                        pre, ext = m.groups()

                        # mask = f"{pre}.mask{end}.{ext}"


                        vh_name = f"{pre}_VH.{ext}"
                        vv_name = f"{pre}_VV.{ext}"
                        # print(vh_name + "\t" + vv_name)
                        data_frame = (
                            os.path.join(dirpath, timeseries_path, vh_name), os.path.join(dirpath, timeseries_path, vv_name)
                            )

                        time_series_data.append(data_frame)

                    # each data point is a list of time series vv + vh pairs with corresponding cropland masks
                    data = (time_series_data, crop_mask_path)
                    folder = os.path.basename(data_dir)

                    if crop_mask_path != '':
                        if edit:
                            if folder == 'test' or folder == 'train':
                                train_metadata.append(data)
                        else:
                            if folder == 'train':
                                # print("Appended to train_metadata\t", data)
                                train_metadata.append(data)
                            elif folder == 'test':
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
    for time_series, mask_name in metadata:
        print(mask_name)
        time_series_stack = []

        # iterate through vv vh image pairs, make a composite, and append them to our list
        for tile_vh, tile_vv in time_series:
            # print(tile_vh + "\n" + tile_vv)
            tif_vh = gdal.Open(tile_vh)
            if(tif_vh):
                print("vh tif found")
                print(tif_vh.RasterXSize)
            # Should prevent the following error
            # ValueError: cannot reshape array of size 524288 into shape (64,64,2)
            comp = str(tif_vh.RasterXSize)
            compositeIsDems = comp != str(dems)
            mockNotInComp = "mock" not in comp
            # if compositeIsDems and mockNotInComp:
            # # if(comp != str(dems) and "mock" not in comp):
            #     # mock is include for the unit tests
            #     print(compositeIsDems, "\t", mockNotInComp)
            #     continue
            tile_vh_array = []
            tile_vv_array = []

            try:
                with gdal_open(tile_vh) as f:
                    tile_vh_array = f.ReadAsArray()
            except FileNotFoundError:
                continue
            try:
                with gdal_open(tile_vv) as f:
                    tile_vv_array = f.ReadAsArray()
            except FileNotFoundError:
                continue

            print("tiling array")
            tile_array = np.stack((tile_vh_array, tile_vv_array), axis=2)
            print("composite length", len(tile_array))
            # if not edit:
            #     if not valid_image(tile_array):
            #         print("ERROR: not valid image")
            #         continue

            x = np.array(tile_array).astype('float32')
            # Clip all x values to a fixed range
            if clip_range:
                min_, max_ = clip_range
                np.clip(x, min_, max_, out=x)
            
            # time_series_stack.stack(tile_array, axis=0)
            time_series_stack.append(x)
        
        # transform our list of timeseries composites into (timesteps, dim, dim, 2)
        print(len(time_series_stack))
        res = np.array(time_series_stack)

        with gdal_open(mask_name) as f:
            mask_array = f.ReadAsArray()
            y = np.array(mask_array).astype('float32')
    
        print("OUTPUT:\t", res)
        yield (res.reshape(output_shape), y.reshape(mask_output_shape))
