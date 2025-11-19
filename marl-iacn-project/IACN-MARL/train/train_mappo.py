# train/train_mappo.py
import os
import sys
import argparse

# 项目根目录绝对路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

try:
    from baselines.mappo.mappo import run_mappo
    from xuance.common import get_configs
    from utils.logger import Logger
    from utils.model_saver import BestModelSaver  # 导入通用工具
    print("✓ 依赖导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

if __name__ == '__main__':
    # 配置文件路径
    config_file = os.path.join(ROOT_DIR, "configs", "smac", "mappo", "3m.yaml")

    if not os.path.exists(config_file):
        print("❌ 配置文件不存在!")
        sys.exit(1)

    try:
        # 读取配置
        configs_dict = get_configs(file_dir=config_file)
        configs = argparse.Namespace(**configs_dict)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    # 设置路径
    configs.log_dir = os.path.join(ROOT_DIR, "logs", "mappo")
    configs.model_dir = os.path.join(ROOT_DIR, "models", "mappo")

    # 创建目录
    os.makedirs(configs.log_dir, exist_ok=True)
    os.makedirs(configs.model_dir, exist_ok=True)

    try:
        # 初始化日志器
        logger = Logger(configs.log_dir, logger_type=configs.logger)
        logger.save_config(configs)
        print("✓ 日志器初始化成功")
    except Exception as e:
        print(f"❌ 日志器初始化失败: {e}")
        sys.exit(1)

    print("🚀 开始MAPPO训练...")

    # 初始化最佳模型保存器（使用通用工具）
    best_saver = BestModelSaver(configs.model_dir, "mappo")

    try:
        # 启动MAPPO训练
        run_mappo(configs, best_saver=best_saver)
        print("✓ MAPPO训练完成")
    except Exception as e:
        print(f"❌ MAPPO训练失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭日志器
        logger.close()

        # 获取最佳模型信息
        best_info = best_saver.get_best_info()
        if best_info['best_model_path']:
            print(f"🏆 最佳模型: {best_info['best_model_path']}")
            print(f"📊 最佳奖励: {best_info['best_reward']:.4f}")