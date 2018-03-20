#!/usr/bin/env sh
tools_path="/home/jbian/deephorizon/caffe/build/tools"
LOG=/home/jbian/deephorizon/models/classification/so_placesvggs_tower/log3/vggs-`date +%Y-%m-%d-%H-%M-%S`.log 
$tools_path/caffe train -solver=/home/jbian/deephorizon/models/classification/so_placesvggs_tower/solver3.proto -weights=/home/jbian/deephorizon/models/classification/so_placesvggs_tower/ZF_faster_rcnn_final3layer1024.caffemodel -gpu 0 2>&1 | tee $LOG