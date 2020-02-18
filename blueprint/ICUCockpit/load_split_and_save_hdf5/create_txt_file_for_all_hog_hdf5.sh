#!/bin/bash 

if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi
FULL_PATH=$1
for pid_data_dir in $FULL_PATH*; do
    ls $pid_data_dir/*hog.hdf5 > $pid_data_dir/all_hdf5.txt
done
