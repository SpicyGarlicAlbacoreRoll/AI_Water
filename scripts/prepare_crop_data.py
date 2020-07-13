# prepare_crop_data.py
# command line sample:
# python3 scripts/prepare_data.py sar/SAR_UT_VV.tif

# import libraries
import sys
import numpy as np
from osgeo import gdal

# rename inputs
RTC_file = sys.argv[1]

# get metadata from RTC file
ds = gdal.Open(RTC_file)
prj = ds.GetProjection()
loc = prj.find("UTM zone ")
zone = prj[loc+9:loc+11]
epsg = prj[-8:-3]
epsg = f"EPSG:{epsg}"

# reproject into Albers Equal Area to get bounding box coordinates in CDL required projection
gdal.Warp('temp_RTC_file.tif', RTC_file, srcSRS=epsg,dstSRS="ESRI:102039")
temp_ds = gdal.Open('temp_RTC_file.tif')
info = gdal.Info(RTC_file)
ulx, xres, xskew, uly, yskew, yres = temp_ds.GetGeoTransform()
lrx = ulx + (temp_ds.RasterXSize * xres)
lry = uly + (temp_ds.RasterYSize * yres)
BBOX = str(ulx)+','+str(uly)+','+str(lrx)+','+str(lry)
BBOX_padded = str(ulx+(3*xres))+','+str(lry+(3*yres))+','+str(lrx+(3*xres))+','+str(uly+(3*yres))

URL = f"https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year=2019&bbox={BBOX_padded}"

print('\nGo to this website and copy the tif file url it gives you: ')
print(URL)
print('\nRun the following command, then check your Downloads folder. It may take up to 15 minutes. ')
print('docker run --rm -v ~/Downloads:/root/Downloads ubuntu:18.04 /bin/bash -c "apt-get update -y && apt-get install -y wget; wget <copied_url.tif> -P ~/Downloads/" ')
print('\nName your file something meaningful, then put it in a meaningful folder.')
print('Use crop_crop.py to crop and reproject the CDL data to the RTC granule of your choice.')
print('Use crop_mask10.py to filter CDL data into "crop" and "not crop" categories. ')
