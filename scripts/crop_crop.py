# crop_crop.py
# command line sample:
# python3 scripts/crop_crop.py sar/SAR_UT_VV.tif CDL_UT/CDL_UT_uncropped.tif CDL_UT/CDL_UT_cropped.tif

# import libraries
import sys
import numpy as np
from osgeo import gdal

# rename inputs
RTC_file = sys.argv[1]
CDL_file = sys.argv[2]
new_CDL_file = sys.argv[3]

# get metadata from RTC file
ds = gdal.Open(RTC_file)
prj = ds.GetProjection()
loc = prj.find("UTM zone ")
zone = prj[loc+9:loc+11]
epsg = prj[-8:-3]
epsg = f"EPSG:{epsg}"

# get bounds
info = gdal.Info(RTC_file)
ulx, xres, xskew, uly, yskew, yres = ds.GetGeoTransform()
lrx = ulx + (ds.RasterXSize * xres)
lry = uly + (ds.RasterYSize * yres)

# project CDL file to match RTC file
gdal.Warp(new_CDL_file, CDL_file, outputBounds=[ulx, uly, lrx, lry], xRes=30, yRes=30, dstSRS=epsg)
print('\nBe sure to check the results in QGIS! :) \n')
