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

    train_metadata, test_metadata = make_timerseries_metadata(dataset)
    # Load the entire dataset into memory
    x_train = []
    y_train = []
    for img, mask in generate_timeseries_from_metadata(train_metadata, clip_range=(0, 2)):
        x_train.append(img)
        y_train.append(mask)

    x_test = []
    y_test = []
    print(test_metadata)
    for img, mask in generate_timeseries_from_metadata(test_metadata, clip_range=(0, 2)):
        x_test.append(img)
        y_test.append(mask)

    # Needed to work with Time Distributed Layers
    # https://stackoverflow.com/questions/58948739/reshaping-images-for-input-into-keras-timedistributed-function
    # train_gen = TimeseriesGenerator()
    # test_gen = TimeseriesGenerator()

    train_iter = train_gen.flow(
        x=x_train, y=y_train, batch_size=16
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

    # for folders in new_test_set
    for dirpath, dirnames, filenames in os.walk(dataset_dir(dataset)):
        for data_dir in dirnames:  # for folders in dataset
            if data_dir == 'test' or data_dir == 'train':
                for idx, (timeseries_path, _, timeseries_filenames) in enumerate(os.walk(os.path.join(dirpath, data_dir))):
                    # Os Walk will create empty _ folder path if folder is empty
                    if(not _):
                        print("FFFFFFFFFFFF", _)
                        time_series_data = []
                        # tile_time_series_data = []
                        timeseries_mask = ""
                        crop_mask_path = ""
                        data = []
                        frames = []
                        print(timeseries_filenames)
                        for idy, name in enumerate(sorted(timeseries_filenames)):
                            # print(name)
                            tile_time_series_data = []
                            crop_mask_file = re.search("mask.tif", name)

                            if crop_mask_file:
                                crop_mask_path = os.path.join(
                                    dirpath, timeseries_path, crop_mask_file.group())

                            # skip to avoid double composite pair
                            m = re.match(TITLE_TIME_SERIES_REGEX, name)
                            if not m:
                                continue

                            # print(name)
                            pre, end, ext = m.groups()
                            # if idx == 0:

                            # Need this for tiled masks
                            mask = f"{pre}_VH{end}.{ext}"

                            vh_name = f"{pre}_VH{end}.{ext}"
                            vv_name = f"{pre}_VV{end}.{ext}"
                            frame = end
                            # print(vh_name)
                            # grab every string with matching tile index for vv and vh tiles, ignoring repeats
                            VV_Tiles = []
                            if frame not in frames:
                                VV_Tiles = [
                                    tileVV for tileVV in sorted(timeseries_filenames)
                                    if re.match(fr"(.*)\_VV{end}\.(tif|tiff)", tileVV)
                                ]
                                frames.append(end)
                            else:
                                break

                            # if idx == 1:
                            #     print(temp)
                            for tileVV in VV_Tiles:
                                tile_time_series_data.append(
                                    (
                                        os.path.join(
                                            dirpath, timeseries_path, tileVV),
                                        os.path.join(
                                            dirpath, timeseries_path, tileVV.replace("VV", "VH"))
                                    )
                                )

                            # os.path.join(dirpath, timeseries_path, vh_name), os.path.join(dirpath, timeseries_path, vv_name)

                            # tuple (vv+vh list, mask)
                            data_frame = (
                                tile_time_series_data, os.path.join(
                                    dirpath, timeseries_path, vh_name)
                            )

                            data.append(data_frame)
                        folder = os.path.basename(data_dir)

                        if crop_mask_path != '':
                            if edit:
                                if folder == 'test' or folder == 'train':
                                    train_metadata.append(data)
                            else:
                                if folder == 'train':
                                    print("TRAINING FOLDER")
                                    train_metadata.append(data)
                                elif folder == 'test':
                                    print("TESTING FOLDER")
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
    # for time stacks           Tuple(List[Tuple(Str, Str)], Str)
    if(len(metadata) == 0):
        print("Empty Object passed")
    
    for time_series_mask_pair in metadata:
        # print(time_series_mask_pair)
        # print("\n\n")
        for time_series, mask in time_series_mask_pair:
            print("MASK:\t", mask)

            print("*********************************************")
            for vv_tile, vh_tile in time_series:
                print("\t", vv_tile, "\n\t", vh_tile)
            
            print("*********************************************\n")
    # for stack in metadata[0]:
    #     print(len(metadata))
    #     print("\n")
    #     mask_name = stack[1]
    #     print("mask_name:\t", mask_name)
    #     # z = input()
    #     #for time stack and corresponding mask              List[Tuple(Str, Str)]
    #     for time_series in stack[0]:
    #         # print(mask_name)
    #         print(time_series)

    #         mask = ''
    #         time_series_stack = []

    #         # for vv and vh tile                Tuple(Str, Str)
    #     # for tile_vh, tile_vv in time_series:
    #         tile_vv = time_series[0]
    #         tile_vh = time_series[1]
    #         tile_vh_as_array = []
    #         tile_vv_as_array = []

    #         tif_vh = gdal.Open(tile_vh)

    #         # Should prevent the following error
    #         # ValueError: cannot reshape array of size 524288 into shape (dem,dem,2)
    #         comp = str(tif_vh.RasterXSize)

    #         # if compositeIsDems and mockNotInComp:
    #         if(comp != str(dems) and "mock" not in comp):
    #             # mock is include for the unit tests
    #             # print(compositeIsDems, "\t", mockNotInComp)
    #             continue


    #         try:
    #             with gdal_open(tile_vh) as f:
    #                 tile_vh_as_array = f.ReadAsArray()
    #         except FileNotFoundError:
    #             continue
    #         try:
    #             with gdal_open(tile_vv) as f:
    #                 tile_vv_as_array = f.ReadAsArray()
    #         except FileNotFoundError:
    #             continue

    #         print("tiling array")
    #         tile_array = np.stack((tile_vh_as_array, tile_vv_as_array), axis=2)
    #         # print("composite length", len(tile_array))
    #         # if not edit:
    #         #     if not valid_image(tile_array):
    #         #         print("ERROR: not valid image")
    #         #         continue

    #         x = np.array(tile_array).astype('float32')
    #         # Clip all x values to a fixed range
    #         if clip_range:
    #             min_, max_ = clip_range
    #             np.clip(x, min_, max_, out=x)

    #         time_series_stack.append(x)

    #         with gdal_open(mask_name) as f:
    #             mask_array = f.ReadAsArray()
    #             y = np.array(mask_array).astype('float32')

    #         print(y.shape)
    #         # y = np.expand_dims(y, axis=0)
    #         print("y expanded shape:\t", y.shape)
    #         yield (np.array(time_series_stack), y.reshape(mask_output_shape))