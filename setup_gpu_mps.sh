#!/bin/bash
systemctl stop gdm3
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -pm 1
# list all available clocks
# nvidia-smi -q -d SUPPORTED_CLOCKS
sudo nvidia-smi -lgc 3120,3120
# sudo nvidia-smi -lgc 1980,1980
sudo nvidia-smi -lmc 10501,10501

nvidia-smi -i 0 -c 3 # -c 3 同 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

# nvidia-settings -a [gpu:0]/GPUPowerMizerMode=1 # set up performance mode
# sudo systemctl stop gdm3