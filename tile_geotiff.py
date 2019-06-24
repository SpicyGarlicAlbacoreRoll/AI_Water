#! /usr/bin/env python3
"""
Rohan Weeden
Helper script for tiling a GeoTiff, creating image labels, and for preparing
data to be used in the network.


## Annoying dependencies:
  * matplotlib
  * gdal

# Usage
First tile the image:
    `$ python3 tile_geotiff.py tile path/to/geo.tiff <size of tiles>`
Then classify the tiles:
    `$ python3 tile_geotiff.py classify path/to/`
"""

import json
import os
import random
import re
from argparse import ArgumentParser
from typing import Tuple

import src.config as config

try:
    from matplotlib import pyplot
    from matplotlib.widgets import Button, RadioButtons
except ImportError:
    pass

try:
    import gdal
except ImportError:
    pass

EXT = "tiff|tif|TIFF|TIF"
FILENAME_REGEX = re.compile(f'.*_ulx_.*\\.(?:{EXT})')


def make_tiles(ifname: str, tile_size: Tuple[int, int]) -> None:
    """Takes a .tiff file and breaks it into smaller .tiff files"""
    img_fpath = os.path.join(config.PROJECT_DIR, 'prep_tiles', ifname)

    if not check_dependencies(('gdal', )):
        return

    datafile = gdal.Open(img_fpath)
    iftitle, ifext = re.match(r'(.*)\.(tiff|tif)', img_fpath).groups()
    step_x, step_y = tile_size

    xsize = datafile.RasterXSize
    ysize = datafile.RasterYSize

    for x in range(0, xsize, step_x):
        for y in range(0, ysize, step_y):
            gdal.Translate(
                f'{iftitle}_ulx_{x}_uly_{y}.{ifext}',
                img_fpath,
                srcWin=[x, y, step_x, step_y],
                format="GTiff"
            )


def interactive_classifier(directory: str) -> None:
    if not check_dependencies(('gdal', 'matplotlib')):
        return

    try:
        with open(os.path.join(directory, 'labels.json'), 'r') as f:
            image_labels = json.load(f)
    except FileNotFoundError:
        image_labels = {}

    for file in os.listdir(directory):
        if not re.match(FILENAME_REGEX, file):
            continue
        current_label = image_labels.get(file)
        if current_label and current_label != 'skip':
            continue

        tif = gdal.Open(os.path.join(directory, file))
        tif_array = tif.ReadAsArray()
        print("Name: ", file)
        # Hack for signaling that the close button was clicked
        close = {"close": False}
        _show_plot(tif_array, file, image_labels, close)

        if close['close']:
            break

        with open(os.path.join(directory, 'labels.json'), 'w') as f:
            json.dump(image_labels, f)


def _show_plot(tif_array, file, image_labels, close):
    pyplot.imshow(tif_array, cmap=pyplot.get_cmap('gray'))
    pyplot.colorbar()
    rax = pyplot.axes([0.05, 0.7, 0.15, 0.15])
    classify_radio = RadioButtons(
        rax, ('Water', 'Not Water', 'Skip', 'Invalid')
    )
    bclose = Button(pyplot.axes([0.05, 0.05, 0.1, 0.075]), 'Close')

    def close_handler(event):
        close['close'] = True
        pyplot.close()

    bclose.on_clicked(close_handler)

    def click_handler(label: str):
        image_labels[file] = label.lower().replace(' ', '_')
        pyplot.close()

    classify_radio.on_clicked(click_handler)

    mng = pyplot.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    pyplot.show()


def prepare_data(directory: str, holdout: float):
    """Moves images to correct directory for binary data,
     calls prepare_mask_data to prepare masked data."""
    try:
        with open(os.path.join(directory, 'labels.json'), 'r') as f:
            image_labels = json.load(f)
    except FileNotFoundError:
        prepare_mask_data(directory, holdout)
        return

    file_names = list(
        filter(
            lambda x: re.match(FILENAME_REGEX, x) is not None,
            os.listdir(directory)
        )
    )

    for file in file_names:
        if file not in image_labels:
            continue

        test_or_train = 'train' if random.random() > holdout else 'test'
        folder = os.path.join(directory, test_or_train)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        os.rename(os.path.join(directory, file), os.path.join(folder, file))


def prepare_mask_data(directory: str, holdout: float) -> None:
    """Renames and moves mask and tile images"""
    TILE_REGEX = re.compile(f"resized(.*)Tile_ulx_([0-9]+)_uly_([0-9]+)\\.({EXT})")

    for file in os.listdir(directory):
        m = re.match(TILE_REGEX, file)
        if not m:
            continue

        pre, num, num2, ext = m.groups()
        name_pre = f"{pre}_{num2}_{num}"
        new_tile_name = f"{name_pre}.tile.{ext}".lower()
        mask_name = f"resizedMASK{pre}Tile_ulx_{num}_uly_{num2}.{ext}"
        new_mask_name = f"{name_pre}.mask.{ext}".lower()

        if not os.path.isfile(os.path.join(directory, mask_name)):
            print(f"Tile: {file} is missing a mask {mask_name}!")
            continue

        test_or_train = 'train' if random.random() > holdout else 'test'

        folder = os.path.join(directory, test_or_train)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        os.rename(
            os.path.join(directory, file), os.path.join(folder, new_tile_name)
        )
        os.rename(
            os.path.join(directory, mask_name),
            os.path.join(folder, new_mask_name)
        )


def check_dependencies(deps: Tuple[str, ...]) -> bool:
    global_vars = globals()
    for dep in deps:
        if dep not in global_vars:
            print(
                f"This function requires {dep}. "
                "Please install it in the current shell and try again."
            )
            return False
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    # make_tiles
    parser_make_tiles = subparsers.add_parser(
        'tile', help='Tile a tiff into many smaller tiffs'
    )
    parser_make_tiles.add_argument("file_name")
    parser_make_tiles.add_argument(
        "tile_size", type=int, help='Side length/height of the square tiles'
    )

    def make_tiles_wrapper(args):
        make_tiles(args.file_name, (args.tile_size, args.tile_size))

    parser_make_tiles.set_defaults(func=make_tiles_wrapper)

    # interactive_classifier
    parser_classifier = subparsers.add_parser(
        'classify', help='Classify tiled images'
    )
    parser_classifier.add_argument("directory")

    def interactive_classifier_wrapper(args):
        interactive_classifier(args.directory)

    parser_classifier.set_defaults(func=interactive_classifier_wrapper)

    # prepare_data
    parser_prepare = subparsers.add_parser(
        'prepare',
        help=
        'Prepare the data directory for use with `ImageGenerator.flow`'
    )
    parser_prepare.add_argument("directory")
    parser_prepare.add_argument(
        "holdout", type=float, help='Proportion of data to use for testing'
    )

    def prepare_data_wrapper(args):
        prepare_data(args.directory, args.holdout)

    parser_prepare.set_defaults(func=prepare_data_wrapper)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
