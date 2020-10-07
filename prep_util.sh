#!/usr/bin/env bash

echo "hello_bash"
i=0
cd prep_tiles
# mkdir testing
# mkdir train
cd ..
TESTING_FILES=()
TRAINING_FILES=()
ls prep_tiles/ | grep 'S.*.tif' | while read -r line ; do
    # echo "$line"
    TILE=$line
    echo $TILE
    /usr/bin/python3 scripts/prepare_data.py tile $TILE 128 && cd prep_tiles && rm $TILE && cd ..
    
    
    cd prep_tiles
    find ./ -maxdepth 1 -iname "*ulx*tif" -exec mv {} tiles \;        
    cd ..
done

ls prep_tiles/ | grep '.*mask.tif' | while read -r line ; do
    MASK=$line
    echo $MASK
    /usr/bin/python3 scripts/prepare_data.py tile $MASK 128
    cd prep_tiles
    find ./ -maxdepth 1 -iname "*ulx*tif" -exec mv {} mask_tiles \; 
    cd ..

done
# echo $vv_vh_files
# python scripts/prepare_data.py tile 