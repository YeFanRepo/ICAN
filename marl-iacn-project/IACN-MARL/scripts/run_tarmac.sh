#!/bin/bash
# 自动切换到项目根目录，确保路径计算100%正确
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# 启动训练（输出日志到文件，方便后续查看）
python train/train_tarmac.py > logs/tarmac/train.log 2>&1 &
echo "TarMAC训练已启动，日志查看：logs/tarmac/train.log"
