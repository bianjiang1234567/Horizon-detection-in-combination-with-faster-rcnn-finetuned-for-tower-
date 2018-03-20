#!/usr/bin/env sh
images_path="/home/jbian/deephorizon/data/images/"
list_file="/home/jbian/deephorizon/data/images/train.txt"
dst_lmdb_file="/home/jbian/deephorizon/data/images/horizon_lmdb"
tools_path="/home/jbian/deephorizon/caffe/build/tools"
rm -rf $dst_lmdb_file
$tools_path/convert_imageset --shuffle $images_path $list_file $dst_lmdb_file