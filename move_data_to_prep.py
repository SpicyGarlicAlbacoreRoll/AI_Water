from tqdm import tqdm
import os
import shutil


def move_subdataset_to_prep(target_dataset, target_subdataset):
    base_dataset_path = f"{os.path.join(os.getcwd(), 'datasets', target_dataset)}"

    subdataset_test = os.path.join(base_dataset_path, "test", target_subdataset)
    subdataset_train = os.path.join(base_dataset_path, "train", target_subdataset)

    if not os.path.isdir(subdataset_test):
        print(f"{subdataset_test} is not a directory")
        return

    if not os.path.isdir(subdataset_train):
        print(f"{subdataset_train} is not a directory")
        return

    destination = os.path.join(os.getcwd(), "prep_tiles", "tiles")

    all_files = [os.path.join(subdataset_test, file) for file in os.listdir(subdataset_test)]
    all_files.extend([os.path.join(subdataset_train, file) for file in os.listdir(subdataset_train)])
    move_all_files(all_files, destination)

def move_all_files(target_files, destination):
    for file in tqdm(target_files):
        shutil.move(file, destination)