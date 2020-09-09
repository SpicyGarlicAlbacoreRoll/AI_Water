import os
import re
import json
from tqdm import tqdm
from typing import Dict, List


"""A script to repair metadata for test/training data
in case the metadata.json for a subdataset in a dataset is unreadable (a user creates a typo editing it)
this script will reads all the data in either the train or test folder of the subdataset and create a new json file
with either a train or test object containing the found file paths. 
After the file is created, the user must copy the test/train object, open the bad metadatafile, and paste
it over the corresponding test/train object entry.

This is useful, as the files can be very large and tracking
down the typo might be very difficult over a remote connection or even on the user's machine"""
def repair_metadata() -> None:
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


    frames = get_tile(files, data_type)


    sub_dataset_name = path_to_target.split("/")[-2]
    file_name = f"{sub_dataset_name}_{data_type}_metadata.json"

    metadata = {}
    with open(file_name, 'r') as fp:
        metadata = json.load(path_to_metadata_target, fp, indent=4)
    
    
    # print(frames[frame_names[0]])

def get_tile(files: List, datatype: str):
    frames = {}
    frame_pattern = re.compile(r"S(.*)\_ulx_(.*)\_uly_(.*)\.(tiff|tif|TIFF|TIF)")

    for file in files:
        m = re.match(frame_pattern, file)
        if not m:
            continue
        _,x,y,_ = m.groups()
        frames[f"ulx_{x}_uly_{y}"] = []
    
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
    return output_frames

def yes_or_no(question) -> bool:
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply == 'y':
            return True
        if reply == 'n':
            return False

repair_metadata()