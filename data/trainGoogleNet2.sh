#!/usr/bin/env sh
LOG=/home/jbian/deephorizon/models/classification/so_places/log/Googlelog2-`date +%Y-%m-%d-%H-%M-%S`.log 
caffe train -solver=/home/jbian/deephorizon/models/classification/so_places/solver2.proto -weights=GoogleNet_SOS.caffemodel -gpu 0 2>&1 | tee $LOG