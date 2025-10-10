#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
nvidia-smi -i 0 -pm 1
sudo nvidia-smi -lgc 1950,1950
sudo nvidia-smi -lmc 9500,9500

nvidia-smi -i 0 -c 3 # -c 3 同 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d

# nvidia-settings -a [gpu:0]/GPUPowerMizerMode=1 # set up performance mode
# sudo systemctl stop gdm3