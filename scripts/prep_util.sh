#!/usr/bin/env bash

echo "hello_bash"
i=0
cd prep_tiles
# mkdir testing
# mkdir train
cd ..
TESTING_FILES=()
TRAINING_FILES=()
ls prep_tiles/ | grep '.*_VV.*.tif' | while read -r line ; do
    # echo "$line"
    VV_TILE=$line
    VH_TILE=${line/VV/'VH'}

    echo $VV_TILE
    echo $VH_TILE
    /usr/bin/python3 scripts/prepare_data.py tile $VV_TILE 512
    /usr/bin/python3 scripts/prepare_data.py tile $VH_TILE 512
    
    
    cd prep_tiles
    if [[ $((i % 2)) == 0 ]]; then
        mv *"ulx"*"tif" 'testing'        
            # TESTING_FILES
    else
        mv *"ulx"*"tif" 'train'
    fi
    cd ..
    ((i++))
done
# echo $vv_vh_files
# python scripts/prepare_data.py tile 