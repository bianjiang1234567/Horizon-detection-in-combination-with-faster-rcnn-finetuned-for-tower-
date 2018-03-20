#!/usr/bin/env sh
images_path="/home/jbian/deephorizon/data/images/"
list_file="/home/jbian/deephorizon/data/images/trainmulti.txt"
dst_lmdb_file="/home/jbian/deephorizon/data/images"
tools_path="/home/jbian/deephorizon/caffe/build/tools"
#rm -rf $dst_lmdb_file
rm -rf train_data_lmdb train_labels_lmdb
$tools_path/convert_imageset --shuffle $images_path $list_file train_data_lmdb train_labels_lmdb 2 