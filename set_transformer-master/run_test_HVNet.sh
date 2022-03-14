#!/bin/sh

export GPU_ID=$1

echo GPU_ID:$GPU_ID

export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python TestHVNet.py $2 $3 $4

# bash run_test_HVNet.sh 1 model_M3_HVNet_5.pth test_data_M3_0.mat cuda
