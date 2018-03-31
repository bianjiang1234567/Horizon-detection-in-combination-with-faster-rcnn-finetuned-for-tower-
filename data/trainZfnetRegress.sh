#!/usr/bin/env sh
tools_path="/home/jbian/deephorizon/caffe/build/tools"
LOG=/home/jbian/deephorizon/models/regression/regularize_so_L1Loss_ZF/log/log-`date +%Y-%m-%d-%H-%M-%S`.log 
$tools_path/caffe train -solver=/home/jbian/deephorizon/models/regression/regularize_so_L1Loss_ZF/solver.proto -weights=/home/jbian/deephorizon/models/regression/regularize_so_L1Loss_ZF/ZF.v2.caffemodel -gpu 0 2>&1 | tee $LOG