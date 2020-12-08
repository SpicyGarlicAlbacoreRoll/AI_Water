from tqdm import tqdm
import os
import shutil
from sys import argv

def move_tiles_to_dest(target_dir: str = "tiles/"):
    prep_tiles_path = f"{os.path.join(os.getcwd(), 'prep_tiles')}"

    if not os.path.isdir(prep_tiles_path):
        print(f"{prep_tiles_path} is not a valid directory")
        return False
    
    target = os.path.join(prep_tiles_path, target_dir)

    if not os.path.isdir(prep_tiles_path):
        print(f"{target} is not a valid directory")
        return False

    os.listdir(prep_tiles_path)

    all_files = [os.path.join(prep_tiles_path, file) for file in os.listdir(prep_tiles_path) 
    if (file.startswith("S") and not file.endswith("VH.tif") and not file.endswith("VV.tif")) 
    or (file.startswith("CDL") and not file.endswith("mask.tif"))]
    move_all_files(all_files, target)

def move_all_files(target_files, destination):
    for file in tqdm(target_files):
        shutil.move(file, destination)

if __name__ == "__main__":
    move_tiles_to_dest(argv[1])

