#!/usr/bin/env bash

TILE_SIZE=${1:256}
i=0

ls prep_tiles/ | grep 'S.*.tif' | while read -r line ; do
    TILE=$line
    echo $TILE
    /usr/bin/python3 scripts/prepare_data.py tile $TILE $TILE_SIZE && cd prep_tiles && rm $TILE && cd ..
    
    /usr/bin/python3 move_prep_tiles.py "tiles/"

done

ls prep_tiles/ | grep '.*mask.tif' | while read -r line ; do
    MASK=$line
    echo $MASK
    /usr/bin/python3 scripts/prepare_data.py tile $MASK $TILE_SIZE
    
    /usr/bin/python3 move_prep_tiles.py "mask_tiles/"

done