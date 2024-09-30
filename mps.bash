#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -c 3 # -c 3 同 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d


nvidia-settings -a [gpu:0]/GPUPowerMizerMode=1 # set up performance mode