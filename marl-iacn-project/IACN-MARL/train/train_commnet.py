# train/train_commnet.py（训练入口，路径绝对化）
import os
import sys
import argparse
from xuance.common import get_configs
# 导入baselines中的CommNet核心函数
from baselines.commnet.commnet import run_commnet

# 项目根目录（IACN-MARL）绝对路径，与logs/models同级
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)  # 加入Python搜索路径，确保能导入baselines

if __name__ == '__main__':
    # 配置文件绝对路径（指向smac/commnet/3m.yaml）
    config_file = os.path.join(ROOT_DIR, "configs", "smac", "commnet", "3m.yaml")

    # 读取配置并转换为Namespace
    configs_dict = get_configs(file_dir=config_file)
    configs = argparse.Namespace(**configs_dict)

    # 强制日志/模型路径（覆盖配置文件，固定到项目根目录下）
    configs.log_dir = os.path.join(ROOT_DIR, "logs", "commnet")
    configs.model_dir = os.path.join(ROOT_DIR, "models", "commnet")

    # 启动训练
    run_commnet(configs)
