#!/bin/sh

export GPU_ID=$1

echo GPU_ID:$GPU_ID

export CUDA_VISIBLE_DEVICES=$GPU_ID
# Activate the relevant virtual environment:
python HVCnet.py $2 $3

# bash run.sh GPU_ID TRAIN_FILE SAVE_MODEL
# bash run.sh 0 train_data_M3_10.mat model_M3_10.pth