import os
import re
import json
from tqdm import tqdm
from typing import Dict, List

def get_tile(files: List, datatype: str):
    frames = {}
    frame_pattern = re.compile(r"S(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")

    for file in files:
        m = re.match(frame_pattern, file)
        if not m:
            continue
        _,x,y,_ = m.groups()
        frames[f"ulx_{x}_uly_{y}"] = []
    
    frame_names = list(set(frames))
    file_pattern = re.compile(r"(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")
    for file in tqdm(files):
        if "VH" in file:
            continue
        m = re.match(frame_pattern, file)
        if not m:
            continue
        _, x, y, _ = m.groups()
        frames[f"ulx_{x}_uly_{y}"].append(file)
        frames[f"ulx_{x}_uly_{y}"].append(file.replace("VV", "VH"))

    output_frames = {datatype: frames}
    return output_frames, frame_names

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply == 'y':
            return True
        if reply == 'n':
            return False

def main():
    basepath = os.getcwd()

    target = input("enter relative directory path to data:\t")
    
    path_to_target = os.path.join(basepath, target)
    if not os.path.isdir(path_to_target):
        print(f"{path_to_target} is not a valid directory")
        return
    
    metadata_target = input("enter relative path and file name of metadata:\t")
    path_to_metadata_target = os.path.join(basepath, metadata_target)
    if not os.path.isdir(path_to_metadata_target):
        print(f"{path_to_metadata_target} is not a valid directory")
        return

    files = [ file for file in os.listdir(path_to_target)]
    training = yes_or_no("Training data?")

    data_type = "test"
    
    if training:
        data_type = "train"
    else:
        data_type = "test"


    frames, frame_names = get_tile(files, data_type)


    sub_dataset_name = path_to_target.split("/")[-2]
    file_name = f"{sub_dataset_name}_{data_type}_metadata.json"

    metadata = {}
    with open(file_name, 'r') as fp:
        metadata = json.load(path_to_metadata_target, fp, indent=4)
    
    
    # print(frames[frame_names[0]])


main()