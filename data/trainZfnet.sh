#!/usr/bin/env sh
LOG=/home/jbian/deephorizon/models/classification/so_placeszf/log/log-`date +%Y-%m-%d-%H-%M-%S`.log 
caffe train -solver=/home/jbian/deephorizon/models/classification/so_placeszf/solver.proto -weights=ZF.v2.caffemodel -gpu 0 2>&1 | tee $LOG