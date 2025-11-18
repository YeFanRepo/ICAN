#!/bin/bash
# scripts/run_mappo.sh（一键启动MAPPO训练）

# 自动切换到项目根目录（IACN-MARL），确保路径计算正确
cd $(dirname $(dirname $(realpath $0)))

# 启动train目录的入口脚本
python train/train_mappo.py

# 可选：输出日志到文件（方便后续查看）
# python train/train_mappo.py > logs/mappo/train.log 2>&1
