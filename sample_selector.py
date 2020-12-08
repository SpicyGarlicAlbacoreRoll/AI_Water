import random
from random import Random
from src.dataset.crop_masked import validate_image
from tqdm import tqdm
import numpy as np
from src.gdal_wrapper import gdal_open
import sys
from sample_class_distribution import find_frequencies
import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List
from math import floor
import sys

def valid_mask(frame_name, mask_dict) -> bool:
    frame_name = f"{frame_name}.tif"
    target_mask_file = mask_dict.get(frame_name)
    if target_mask_file == None:
        
        return False
    
    output = []
    target_mask_path = os.path.join(os.getcwd(), "prep_tiles", "mask_tiles", target_mask_file)
    try:
        with gdal_open(target_mask_path) as f:
            output = f.ReadAsArray()
    except FileNotFoundError:
        sys.exit(f"File {target_mask_file} not found!")
    
    output_arr = np.asarray(output).flatten()
    image_size = len(output_arr)
    unique_classes, unique_count = np.unique(output_arr, return_counts=True)
    class_dict = dict(zip(unique_classes, unique_count))
    frequency_dict = find_frequencies(class_dict, unique_classes, image_size)
    
   
    if frequency_dict.get("0") == None:
        return True

    # ignore cropmask files that are primarily one class (background, qgis no-data value)
    if frequency_dict.get("0") > 0.15 and frequency_dict.get("0") < 0.75:
        return True

    return False

def get_tiles(files: List) -> (Dict, List):
    frames = {}
    frame_pattern = re.compile(r"S(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")

    for file in files:
        m = re.match(frame_pattern, file)
        if not m:
            continue
        _, x, y, _ = m.groups()
        frames[f"ulx_{x}_uly_{y}"] = []

    # remove duplicates
    frame_names = list(set(frames))
    # frames.sort()

    file_pattern = re.compile(r"(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")
    for file in files:
        if "VH" in file:
            continue
        m = re.match(frame_pattern, file)
        if not m:
            continue
        _, x, y, _ = m.groups()
        frames[f"ulx_{x}_uly_{y}"].append((file,file.replace("VV", "VH")))

    return frames, frame_names

"""Takes tiled data files frop prep_tiles/tiles/ and randomly splits the time series samples,
90% going to training data and 10% going to testing data (depending on the tile size this might take a while). 
This function will create (int the current working directory) a dataset folder with train and test folders 
both containing a folder with the same name as the dataset. 

A metadata file will also be created, and should be placed in the 
root directory of a dataset using the subdataset created by this function."""
def create_sample_split() -> None:
    state = input("Enter state acronym (IE: AK, WA, OR):\t")
    year = input("Enter year (IE 2017, 2020, 2019):\t")

    dir = f'{state}_{year}'
    print (dir)

    if dir == "_":
        return

    dir_path = f"{os.getcwd()}/{dir}"

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        os.mkdir(f"{dir_path}/test")
        os.mkdir(f"{dir_path}/test/{dir}")
        os.mkdir(f"{dir_path}/train")   
        os.mkdir(f"{dir_path}/train/{dir}") 


    files = []
    files = [
        file 
            for file in 
                os.listdir(f"{os.getcwd()}/prep_tiles/tiles") 
                if os.path.basename(file).startswith("S")
    ]

    frames, frame_names = get_tiles(files)

    test_split = floor((len(frame_names) * .10))
    frame_names.sort()
    
    test_data_frame_names = Random(64).sample(frame_names, test_split)
    print("Splitting test/training data...")
    frame_names = [x for x in frame_names if x not in test_data_frame_names]
    print(f"{len(test_data_frame_names)} / {len(frame_names)} = {len(test_data_frame_names) / len(frame_names)}")
    print("Validating and moving data for...")
    print("Test data")
    updated_frames = {}
    updated_frames["test"] = {}
    updated_frames["train"] = {}
    prep_tiles_path = os.path.join(os.getcwd(), "prep_tiles/")
    mask_files = os.listdir(os.path.join(os.getcwd(), "prep_tiles", "mask_tiles"))
    mask_dict = dict(zip(["_".join(mask_file_name.split("_")[4:]) for mask_file_name in mask_files], mask_files))
    for test_frame_name in tqdm(test_data_frame_names):
        updated_frames["test"][test_frame_name] = []
        if not valid_mask(test_frame_name, mask_dict):
            updated_frames["test"].pop(test_frame_name, None)
            continue
        # if frames[test_frame_name]
        for vv, vh in frames[test_frame_name]:
            if not os.path.isfile(os.path.join(prep_tiles_path,"tiles/", vv)) or not os.path.isfile(os.path.join(prep_tiles_path,"tiles/", vh)):
                continue
            if not validate_image(prep_tiles_path, "tiles/", vv):
                continue
            # vv_file_name = Path(vv).name
            # vh_file_name = Path(vh).name
            


            shutil.move(os.path.join(prep_tiles_path, "tiles/", vv), f"{dir_path}/test/{dir}")
            vv = f"test/{dir}/{vv}"
            updated_frames["test"][test_frame_name].append(vv)

            shutil.move(os.path.join(prep_tiles_path, "tiles/", vh), f"{dir_path}/test/{dir}")
            vh = f"test/{dir}/{vh}"
            updated_frames["test"][test_frame_name].append(vh)
        
        if len(updated_frames["test"][test_frame_name]) <= 1:
            updated_frames["test"].pop(test_frame_name, None)
        
        else:
            temp = updated_frames['test'][test_frame_name]
            temp.sort()
            composite_temp = [(tileVH, tileVV) for tileVH, tileVV in zip(temp[0::2], temp[1::2])]

            # in cases where S1A and S1B are in the same list they are sorted by dates
            composite_temp.sort(key=lambda composite: composite[0].split("_")[1:])
            updated_frames['test'][test_frame_name] = composite_temp

    print("Train data")
    for frame_name in tqdm(frame_names):
        updated_frames["train"][frame_name] = []
        if not valid_mask(frame_name, mask_dict):
            updated_frames["train"].pop(frame_name, None)
            continue
        for vv, vh in frames[frame_name]:
            if not os.path.isfile(os.path.join(prep_tiles_path,"tiles/", vv)) or not os.path.isfile(os.path.join(prep_tiles_path,"tiles/", vh)):
                continue
            if not validate_image(prep_tiles_path, "tiles/", vv):
                continue

            # vv_file_name = Path(vv).name
            # vh_file_name = Path(vh).name
            
            shutil.move(os.path.join(prep_tiles_path,"tiles/",vv), f"{dir_path}/train/{dir}")
            # vv = f"train/{dir}/{vv_file_name}"
            vv = f"train/{dir}/{vv}"
            updated_frames["train"][frame_name].append(vv)

            shutil.move(os.path.join(prep_tiles_path,"tiles/",vh), f"{dir_path}/train/{dir}")
            # vh = f"train/{dir}/{vh_file_name}"
            vh = f"train/{dir}/{vh}"
            updated_frames["train"][frame_name].append(vh)

        if len(updated_frames["train"][frame_name]) <= 1:
            updated_frames["train"].pop(frame_name, None)
        else:
            # sorting ahead of time speeds up model
            temp = updated_frames['train'][frame_name]
            temp.sort()
            composite_temp = [(tileVH, tileVV) for tileVH, tileVV in zip(temp[0::2], temp[1::2])]

            # in cases where S1A and S1B are in the same list they are sorted by dates
            composite_temp.sort(key=lambda composite: composite[0].split("_")[1:])
            updated_frames['train'][frame_name] = composite_temp

    print("Training data finished")
    print("serializing metadata...")
    with open(f"{dir_path}/{dir}_metadata.json", 'w') as fp:
        json.dump(updated_frames, fp, indent=4)
    print(f"metadata serialized to:\t{dir_path}/{dir}_metadata.json")

create_sample_split()