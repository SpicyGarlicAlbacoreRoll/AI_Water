# https://gis.stackexchange.com/a/352944
# fix negative resolution

for file in *.tif; 
    # adjust NS resolution 
    do echo "$file"; gdalwarp -t_srs EPSG:4326 -overwrite "$file" "mosaic/$(basename "$file" .tif).tif"
done
