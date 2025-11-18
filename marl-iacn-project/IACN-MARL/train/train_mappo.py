# train/train_mappo.py（训练入口）
import os
import sys
import argparse  # 必须导入argparse，避免NameError

# 把项目根目录（IACN-MARL）加入Python路径，确保能导入baselines
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入baselines中的MAPPO核心函数
from baselines.mappo.mappo import run_mappo
from xuance.common import get_configs

if __name__ == '__main__':
    # 配置文件路径：基于项目根目录（train的上级是根目录）
    config_path = "../configs/smac/mappo/3m.yaml"
    # 转为绝对路径，避免运行目录影响
    abs_config_path = os.path.join(root_dir, config_path.lstrip("../"))  # 移除../，用root_dir拼接

    # 读取配置
    configs_dict = get_configs(file_dir=abs_config_path)
    configs = argparse.Namespace(**configs_dict)

    # 强制统一日志/模型路径（覆盖配置文件中的相对路径，避免混乱）
    configs.log_dir = os.path.join(root_dir, "logs/mappo/")
    configs.model_dir = os.path.join(root_dir, "models/mappo/")

    # 调用MAPPO核心训练函数
    run_mappo(configs)
