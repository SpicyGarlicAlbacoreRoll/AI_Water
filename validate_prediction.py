from typing import Dict, List, Tuple
from src.dataset.common import dataset_dir
from src.gdal_wrapper import gdal_open
import os
import sys
from tqdm import tqdm
from main import save_img
import numpy as np

def find_mask(prediction_filename: str, subdataset_prefix: str, dataset_path: str):
    mask_filename = prediction_filename.replace("prediction", "mask")
    
    subdataset_prefix = "_".join(mask_filename.split("_")[1:3])

    mask_path = os.path.join(dataset_path, "masks", f"{subdataset_prefix}_masks", mask_filename)

    return mask_path

def open_full_ground_truth_mosaic(subdataset_prefix: str, dataset_path: str):
    file_path_name = f"CDL_{subdataset_prefix}_mask.tif"
    datafile = gdal.Open(file_path_name)

    return datafile


def get_predictions_dirs(prediction_dir: str):
    prediction_dir_path = os.path.join(os.getcwd(), "predictions", prediction_dir)

    files_and_dirs = os.listdir(prediction_dir_path)

    return [folder for folder in files_and_dirs if not folder.endswith(".json")]

def compare_pred_mask_acc(pred_mask_pairs: Tuple[str, str]):

    prediction_fpath =  pred_mask_pairs[0]
    ground_truth_fpath = pred_mask_pairs[1]
    
    prediction = []
    ground_truth = []

    try:
        with gdal_open(prediction_fpath) as f:
            prediction = f.ReadAsArray()
    except FileNotFoundError:
        sys.exit(f"File {target_mask_file} not found!")
    
    try:
        with gdal_open(ground_truth_fpath) as f:
            ground_truth = f.ReadAsArray()
    except FileNotFoundError:
        sys.exit(f"File {target_mask_file} not found!")
    
    prediction = np.array(prediction).astype('uint8').reshape(1024, 1024, 1)
    ground_truth = np.array(ground_truth).astype('uint8').reshape(1024, 1024, 1).flatten()

    output = np.zeros((1024, 1024, 1)).astype('float32').flatten()

    for i, val in enumerate(prediction.flatten()):
        # if the model predicted a crop area accurate
        if ground_truth[i] == val:
            output[i] = 1
        # If the model predicted a crop when there was none there
        elif ground_truth[i] != val and val == 1:
            output[i] = 255
        elif ground_truth[i] != val:
            output[i] = 128
    
    vals = np.unique(output)
    total_pixels = 1024.0 ** 2

    freq_dict = {}
    for val in vals:
        freq_dict[val] = np.count_nonzero(output == val) / total_pixels
    
    output = output.reshape(1024, 1024, 1)
    # print(freq_dict)

    return output, freq_dict
    # for i, val in enumerate(output):
        # output[:, :, i][prediction[:, :, i] == ground_truth[:,:,i]] = 1
        # output[:, :, i][prediction[:, :, i] == 0 and ground_truth[:, :, i] == 1] = 255


def main(target_prediction_dir_name: str):
    # target_prediction_dir_name = 

    dir_path = os.path.join(os.getcwd(), "predictions", target_prediction_dir_name)
    dirs = get_predictions_dirs(target_prediction_dir_name)

    pred_mask_pairs = {}
    for folder in dirs:
        pred_mask_pairs[folder] = []
        pred_files = os.listdir(os.path.join(os.getcwd(), "predictions/", target_prediction_dir_name, folder))
        for file in pred_files:
            f_path = os.path.join(dir_path, folder, file)
            pred_mask_pairs[folder].append((f_path, find_mask(file, folder, dataset_dir("WA_2018_1024"))))

    # print(pred_mask_pairs[dirs[-1]])

    if not os.path.isdir(os.path.join(os.getcwd(), "prediction_validation")):
        os.mkdir(prediction_validation)

    if not os.path.isdir(os.path.join(os.getcwd(), "prediction_validation", target_prediction_dir_name)):
        os.mkdir(os.path.join(os.getcwd(), "prediction_validation", target_prediction_dir_name))

    subdataset = list(pred_mask_pairs)

    for dataset in subdataset:
        subdataset_path = os.path.join(os.getcwd(), "prediction_validation", target_prediction_dir_name, dataset)
        
        if not os.path.isdir(os.path.join(os.getcwd(), "prediction_validation", target_prediction_dir_name, dataset)):
            os.mkdir(subdataset_path)

        subdataset_files = pred_mask_pairs[dataset]
        
        # https://stackoverflow.com/questions/48846876/how-to-mosaic-arrays-using-numpy
        subdataset_mosaic = open_full_ground_truth_mosaic(dataset, subdataset_path)
        xsize = datafile.RasterXSize
        ysize = datafile.RasterYSize
        # band = datafile.GetRasterBand(1)
        # freq_validation_mosaic = np.zeros((len(subdataset_files), 1024, 1024))
        for tile_idx, pred_mask_pair_fpaths in tqdm(enumerate(subdataset_files)):
            # print(pred_mask_pair_fpaths)
            data, freq_dict = compare_pred_mask_acc(pred_mask_pair_fpaths)
            fpath = pred_mask_pair_fpaths[0]
            path_to_freq_validation = os.path.join(subdataset_path, fpath.split("/")[-1].replace("prediction", "pixelValidation"))
            # print(path_to_freq_validation)
            save_img(path_to_freq_validation, fpath, data, dem=1024)
            # freq_validation_mosaic[tile_idx, :, :, :] = data
        
        


if __name__ == "__main__":
    main(sys.argv[1])