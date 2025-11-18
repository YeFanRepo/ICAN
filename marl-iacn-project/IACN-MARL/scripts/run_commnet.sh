#!/bin/bash
# 自动切换到项目根目录（IACN-MARL），确保路径计算100%正确
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# 启动训练，日志输出到logs/commnet/train.log（后台运行，方便查看）
python train/train_commnet.py > logs/commnet/train.log 2>&1 &
echo "CommNet训练已启动，日志查看：logs/commnet/train.log"
