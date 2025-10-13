# 恢复应用时钟（取消锁频）
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
# 或者一把梭：sudo nvidia-smi -i 0 -rac   # reset application clocks

# 关闭持久化与独占
sudo nvidia-smi -i 0 -pm 0
sudo nvidia-smi -i 0 -c 0   # DEFAULT

# 关闭 MPS 守护
echo quit | nvidia-cuda-mps-control || true
rm -rf /tmp/nvidia-mps 2>/dev/null || true

# 恢复 PowerMizer（需要 nvidia-settings 可用）
# nvidia-settings -a "[gpu:0]/GPUPowerMizerMode=0" || true   # 0=Adaptive