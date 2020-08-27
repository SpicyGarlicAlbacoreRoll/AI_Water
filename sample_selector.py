import random
from random import Random
import os
import re
from typing import Dict, List
from math import floor

def get_tiles(files: List) -> (Dict, List):
    frames = {}
    frame_pattern = re.compile(r"(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")

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
    state = ''
    year = ''

    for file in os.listdir(f"{os.getcwd()}/prep_tiles"):
        if file.split("_")[0] == 'CDL' and file.endswith("mask.tif"):
            # example: CDL_WA_2018_mask.tif
            state = file.split("_")[1]
            year = file.split("_")[2]
            break

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

    for file in os.listdir(f"{os.getcwd()}/prep_tiles/train"):
        files.append(f"{os.getcwd()}/prep_tiles/train/{dir}/{file}")
    for file in os.listdir(f"{os.getcwd()}/prep_tiles/testing"):
        files.append(f"{os.getcwd()}/prep_tiles/testing/{dir}/{file}")

    print(files[0])

    frames, frame_names = get_tiles(files)

    test_split = floor((len(frame_names) * .10))
    frame_names.sort()
    
    test_data_frame_names = Random(64).sample(frame_names, test_split)


    print(f"{len(test_data_frame_names)} / {len(frame_names)} = {len(test_data_frame_names) / len(frame_names)}")

    for test_frame_name in test_data_frame_names:
        for vv, vh in frames[test_frame_name]:
            print(f"{vv} {vh}")
    # print(test_data_frame_names)

    # print(frames[frame_names[0]])

    # stacked_frames = []
    # print(len(frames))
    # for idx, frame in enumerate(frames):
    #     r = re.compile(fr"(.*){frame}\.(tiff|tif|TIFF|TIF)")
    #     stack = list(filter(r.match, files))
    #     files = [x for x in files if x not in stack]
    #     stacked_frames.append((stack))
    #     print(idx)
    
    # print(stacked_frames[0])
    # print(frames[-1])
    # if not os.path.exists('')

    # re.match()



create_sample_split()