#!/bin/sh

export GPU_ID=$1

echo GPU_ID:$GPU_ID

export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python TestHVCNet.py $2 $3 $4

# bash run_test.sh 1 model_M3_10.pth test_data_M3_0.mat cuda
