#!/usr/bin/env sh
#images_path="/home/jbian/deephorizon/data/images/"
#list_file="/home/jbian/deephorizon/data/images/trainmulti.txt"
dst_lmdb_file="/home/jbian/deephorizon/data/train_data_lmdb3"
tools_path="/home/jbian/deephorizon/caffe/build/tools"
#rm -rf $dst_lmdb_file
rm -rf mean3.binaryproto 
$tools_path/compute_image_mean $dst_lmdb_file  mean3.binaryproto 