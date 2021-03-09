import numpy as np
import sys
from src.gdal_wrapper import gdal_open
import os
import json
from tqdm import tqdm

def read_file(file_name):
    target_file_path = os.path.join(os.getcwd(), "prep_tiles", file_name)
    output = []
    try:
        with gdal_open(target_file_path) as f:
            output = f.ReadAsArray()
    except FileNotFoundError:
        sys.exit(f"File {target_file_path} not found!")
    
    return output

def find_frequencies(classes_dict, unique_classes, image_size):
    class_frequencies = [float(classes_dict[key] / image_size) for key in unique_classes]
    class_frequency_dict = {}
    
    class_frequency_dict = dict(zip([str(unique_class) for unique_class in unique_classes], class_frequencies))
    return class_frequency_dict
    

def sample_class_distributions():
    # file_name = input("Enter file name to be read from prep_tiles")
    file_name = "CDL_WA_2018_mask.tif"
    data = np.nan_to_num(np.asarray(read_file(file_name)).flatten())
    image_size = len(data)
    print(f"Counting unique classes for {file_name}")
    # get a list of unique classes in image, and the number of occurrences
    unique_classes, counts = np.unique(data, return_counts=True)
    # create a dictionary that links each unique value to it's occurrences in the image
    class_dict = dict(zip(unique_classes, counts))
    
    class_frequency_dict = find_frequencies(class_dict, unique_classes, image_size)
    class_frequency_dict["mask_file"] = file_name
    print("\nclass frequencies:\n")
    for key in unique_classes:
        print(f"class: {key}\n\t{class_frequency_dict[str(key)]}\n")
    
    
    with open(f"{file_name.replace('tif', '')}_frequency.json", 'w') as f:
        json.dump(class_frequency_dict, f, sort_keys=True, indent=4)

# sample_class_distributions()