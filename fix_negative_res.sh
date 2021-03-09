# https://gis.stackexchange.com/a/352944
# fix negative resolution

for file in *.tif; 
    
    # https://gis.stackexchange.com/questions/108673/using-gdal-command-line-to-copy-projections
    PROJ=$(gdalsrsinfo -o wkt $file)
    do echo "$PROJ"
    # adjust NS resolution 
    # do echo "$file"; gdalwarp -t_srs $PROJ -overwrite "$file" "mosaic/$(basename "$file" .tif).tif"
done
