# crop_filter
# command line format:
# python3 scripts/crop_filter.py CDL_UT/CDL_UT_UTM12_original.tif CDL_UT_crops.tif

# import libraries
import sys
import numpy as np
from osgeo import gdal

CDL_file = sys.argv[1]
new_CDL_file = sys.argv[2]

# not all of this is necessary for the script (names) but it's nice to have,
# in case you want to print crop type statistics
crop_ind = np.array([1,2,3,4,5,10,23,24,26,36])
crops = ["corn","cotton","rice","sorghum","soybeans","peanuts","spring_wheat","winter_wheat","double_crop","alfalfa"]

# open the CDL file and make sure the data is a numpy array
ds = gdal.Open(CDL_file)
ds_array = ds.ReadAsArray()
ds_array = np.array(ds_array)
row, col = ds_array.shape

print('total pixels: ', row*col)
count = 0
for c in range(len(crop_ind)):
    print(crops[c],': ',np.count_nonzero(ds_array == crop_ind[c]))
    count = count + np.count_nonzero(ds_array == crop_ind[c])
print('total crops: ',count)


for c in range(len(crop_ind)):
    ds_array[ds_array==crop_ind[c]]=1
ds_array[ds_array!=1]=0

unique, counts = np.unique(ds_array, return_counts=True)
dictionary = dict(zip(unique, counts))
print(dictionary)

# make a copy of the old file, and put the new array values in it
driver = ds.GetDriver()
out = driver.Create(new_CDL_file,col, row, 1)
out_band = out.GetRasterBand(1)
out_band.WriteArray(ds_array)
out.SetGeoTransform(ds.GetGeoTransform())
out.SetProjection(ds.GetProjection())

print('\nZeros indicate no crop, ones indicate one of the top ten crop types: ')
for a in range(10):
    print(a+1,': ',crops[a])
print('')
