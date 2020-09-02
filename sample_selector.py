import random
from random import Random
import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List
from math import floor

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

def create_sample_split():
    state = input("Enter state acronym (IE: AK, WA, OR):\t")
    year = input("Enter year (IE 2017, 2020, 2019):\t")

    # for file in os.listdir(f"{os.getcwd()}/prep_tiles"):
    #     if file.split("_")[0] == 'CDL' and file.endswith("mask.tif"):
    #         # example: CDL_WA_2018_mask.tif
    #         state = file.split("_")[1]
    #         year = file.split("_")[2]
    #         break
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
                os.listdir(f"{os.getcwd()}/prep_tiles/train") 
                if os.path.basename(file).startswith("S")
    ]
    # for file in os.listdir(f"{os.getcwd()}/prep_tiles/train"):
    #     files.append(f"{os.getcwd()}/prep_tiles/train/{file}")
    # for file in os.listdir(f"{os.getcwd()}/prep_tiles/testing"):
    #     files.append(f"{os.getcwd()}/prep_tiles/testing/{file}")

    print(files[0])

    frames, frame_names = get_tiles(files)

    test_split = floor((len(frame_names) * .10))
    frame_names.sort()
    
    test_data_frame_names = Random(64).sample(frame_names, test_split)
    print("Splitting test/training data...")
    frame_names = [x for x in frame_names if x not in test_data_frame_names]
    print(f"{len(test_data_frame_names)} / {len(frame_names)} = {len(test_data_frame_names) / len(frame_names)}")
    print("Moving data...")

    updated_frames = {}
    updated_frames["test"] = {}
    updated_frames["train"] = {}
    prep_tiles_path = os.path.join(os.getcwd(), "prep_tiles/")
    for test_frame_name in test_data_frame_names:
        updated_frames["test"][test_frame_name] = []
        for vv, vh in frames[test_frame_name]:
            if not os.path.isfile(os.path.join(prep_tiles_path,"train/", vv)) or not os.path.isfile(os.path.join(prep_tiles_path,"train/", vh)):
                continue
            
            # vv_file_name = Path(vv).name
            # vh_file_name = Path(vh).name

            shutil.move(os.path.join(prep_tiles_path, "train/", vv), f"{dir_path}/test/{dir}")
            vv = f"test/{dir}/{vv}"
            updated_frames["test"][test_frame_name].append(vv)

            shutil.move(os.path.join(prep_tiles_path, "train/", vh), f"{dir_path}/test/{dir}")
            vh = f"test/{dir}/{vh}"
            updated_frames["test"][test_frame_name].append(vh)

    print("Test split data finished")
    
    for frame_name in frame_names:
        updated_frames["train"][frame_name] = []
        for vv, vh in frames[frame_name]:
            if not os.path.isfile(os.path.join(prep_tiles_path,"train/", vv)) or not os.path.isfile(os.path.join(prep_tiles_path,"train/", vh)):
                continue
            vv_file_name = Path(vv).name
            vh_file_name = Path(vh).name
            
            shutil.move(os.path.join(prep_tiles_path,"train/",vv), f"{dir_path}/train/{dir}")
            vv = f"train/{dir}/{vv_file_name}"
            updated_frames["train"][frame_name].append(vv)

            shutil.move(os.path.join(prep_tiles_path,"train/",vh), f"{dir_path}/train/{dir}")
            vh = f"train/{dir}/{vh_file_name}"
            updated_frames["train"][frame_name].append(vh)

    print("Training data finished")
    print("serializing metadata...")
    with open(f"{dir_path}/{dir}_metadata.json", 'w') as fp:
        json.dump(updated_frames, fp, indent=4)
    print(f"metadata serialized to:\t{dir_path}/{dir}_metadata.json")

create_sample_split()