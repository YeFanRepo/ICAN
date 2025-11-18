#!/bin/bash
# scripts/run_qmix.sh（一键启动QMIX训练）

# 关键：强制切换到项目根目录（IACN-MARL）运行，避免路径错误
cd $(dirname $(dirname $(realpath $0)))  # 这行代码会自动切换到 IACN-MARL 目录

# 启动train目录的训练入口脚本
python train/train_qmix.py

# 可选：添加日志输出到文件（方便后续查看）
# python train/train_qmix.py > logs/qmix/train.log 2>&1
