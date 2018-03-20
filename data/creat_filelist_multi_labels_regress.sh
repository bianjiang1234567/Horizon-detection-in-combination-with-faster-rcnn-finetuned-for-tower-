#!/usr/bin/env sh
images_path="/home/jbian/deephorizon/data/images2/"
list_file="/home/jbian/deephorizon/data/images2/TrainmultiRegression.txt"
#dst_lmdb_file="/home/jbian/deephorizon/data/images"
tools_path="/home/jbian/deephorizon/caffe/build/tools"
#rm -rf $dst_lmdb_file
rm -rf train_data_lmdb3 train_labels_lmdb3
$tools_path/convert_imageset --shuffle --resize_height=224 --resize_width=224 $images_path $list_file train_data_lmdb3 train_labels_lmdb3 4 