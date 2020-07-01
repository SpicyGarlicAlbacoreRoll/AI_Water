# crop_mask10.py
# command line format:
# python3 location/crop_mask10.py CDL_input.tif CDL_output.tif
# command line example:
# python3 scripts/crop_mask10.py CDL_UT/CDL_UT_UTM12_original.tif CDL_UT_crops.tif

# import libraries
import sys
import numpy as np
from osgeo import gdal

CDL_file = sys.argv[1]
new_CDL_file = sys.argv[2]

# crop indices
crop_ind = np.array([1,2,3,4,5,10,23,24,26,36])
crops = ["corn","cotton","rice","sorghum","soybeans","peanuts","spring_wheat","winter_wheat","double_crop","alfalfa"]

# open the CDL file and make sure the data is a numpy array
ds = gdal.Open(CDL_file)
ds_array = ds.ReadAsArray()
ds_array = np.array(ds_array)
row, col = ds_array.shape

# display the crop amounts
count = 0
print('\nCROPS:')
for c in range(len(crop_ind)):
    print(crops[c],': ',np.count_nonzero(ds_array == crop_ind[c]))
    count = count + np.count_nonzero(ds_array == crop_ind[c])
print('\ntotal non-crops: ',row*col - count)
print('total crops: ',count)
print('total pixels: ', row*col)

# change all top ten crop types to ones, and set the rest to zeros
for c in range(len(crop_ind)):
    ds_array[ds_array==crop_ind[c]]=1
ds_array[ds_array!=1]=0

unique, counts = np.unique(ds_array, return_counts=True)
dictionary = dict(zip(unique, counts))
print('Zeros indicate no crop, ones indicate one of the top ten crop types: ')
print(dictionary)
print('Crop pixel amounts should match.')

# make a copy of the old file, and put the new array values in it
driver = ds.GetDriver()
out = driver.Create(new_CDL_file,col, row, 1)
out_band = out.GetRasterBand(1)
out_band.WriteArray(ds_array)
out.SetGeoTransform(ds.GetGeoTransform())
out.SetProjection(ds.GetProjection())
