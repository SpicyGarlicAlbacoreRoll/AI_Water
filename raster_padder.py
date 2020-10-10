import gdal
import os
import subprocess
from gdalconst import GA_ReadOnly

# original author: j08lue
# https://gis.stackexchange.com/a/104367

# returns the minx, miny, maxx, and maxy of a given cropmask
def getMinMax(src):
    data = gdal.Open(src, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    return (minx, miny, maxx, maxy)
    data = None

def findFilePaths():
    dir_path = os.getcwd()
    files_and_dirs = os.listdir(dir_path)
    files = [os.path.join(dir_path, f) for f in files_and_dirs if os.path.isfile(os.path.join(dir_path, f)) and f.startswith("S")]
    return files

def findMask():
    files_and_dirs = [f for f in os.listdir(os.getcwd()) if f.endswith("mask.tif")]
    return os.path.join(os.getcwd(), files_and_dirs[0])

minMax = getMinMax(findMask())
print(minMax)

minx, miny, maxx, maxy = minMax

# subprocess.call(['gdalinfo','CDL_WA_2018_mask.tif'])

file_paths = findFilePaths()
print(file_paths[0])

for file in file_paths:
    subprocess.call(['gdalwarp', '-te', str(minx),str(miny), str(maxx), str(maxy), file, os.path.join(os.getcwd(), "padded", os.path.split(file)[1])])