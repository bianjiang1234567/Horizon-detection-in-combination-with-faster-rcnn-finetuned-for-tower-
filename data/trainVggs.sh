#!/usr/bin/env sh
LOG=/home/jbian/deephorizon/models/classification/so_placesvggs/log/vggs-`date +%Y-%m-%d-%H-%M-%S`.log 
caffe train -solver=/home/jbian/deephorizon/models/classification/so_placesvggs/solver.proto -weights=VGG_CNN_S.caffemodel -gpu 0 2>&1 | tee $LOG