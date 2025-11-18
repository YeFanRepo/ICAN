# train/train_tarmac.py（训练入口，固定路径逻辑）
import os
import sys
import argparse  # 导入argparse，避免NameError

# 强制添加项目根目录（IACN-MARL）到Python路径，确保能导入baselines模块
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入baselines中的TarMAC核心训练函数
from baselines.tarmac.tarmac import run_tarmac
from xuance.common import get_configs

if __name__ == '__main__':
    # 配置文件路径：基于项目根目录（train的上级是根目录）
    config_path = "../configs/smac/tarmac/3m.yaml"
    # 转为绝对路径（避免运行目录影响，确保找到配置文件）
    abs_config_path = os.path.join(root_dir, config_path.lstrip("../"))  # 移除../，用根目录拼接

    # 读取配置文件
    configs_dict = get_configs(file_dir=abs_config_path)
    configs = argparse.Namespace(**configs_dict)

    # 强制统一日志/模型保存路径（覆盖配置文件中的相对路径，避免混乱）
    configs.log_dir = os.path.join(root_dir, "logs/tarmac/")
    configs.model_dir = os.path.join(root_dir, "models/tarmac/")

    # 调用TarMAC核心训练函数
    run_tarmac(configs)
