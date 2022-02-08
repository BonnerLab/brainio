#!/bin/bash
directory="$SHARED_DATASETS/ade20k"
mkdir -p $directory

cd $directory || exit

# get the link from http://groups.csail.mit.edu/vision/datasets/ADE20K/request_data/account.php after signing in (only valid for 4 h)
curl -o "$directory/images.zip" "https://groups.csail.mit.edu/vision/datasets/ADE20K/syml/rajmagesh_56909f7f.zip"
unzip "images.zip"
rm "images.zip"
