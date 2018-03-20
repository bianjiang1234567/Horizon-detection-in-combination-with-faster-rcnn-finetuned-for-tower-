#!/usr/bin/env sh
tools_path="/home/jbian/deephorizon/caffe/build/tools"
LOG=/home/jbian/deephorizon/models/regression/regularize_so_l2_vggs/log/vggs-`date +%Y-%m-%d-%H-%M-%S`.log 
$tools_path/caffe train -solver=/home/jbian/deephorizon/models/regression/regularize_so_l2_vggs/solver.proto -weights=VGG_CNN_S.caffemodel -gpu 0 2>&1 | tee $LOG