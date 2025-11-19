# train/train_commnet.py
import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

try:
    from baselines.commnet.commnet import run_commnet
    from xuance.common import get_configs
    from utils.logger import Logger
    from utils.model_saver import BestModelSaver

    print("✓ 依赖导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

if __name__ == '__main__':
    config_file = os.path.join(ROOT_DIR, "configs", "smac", "commnet", "3m.yaml")

    if not os.path.exists(config_file):
        print("❌ 配置文件不存在!")
        sys.exit(1)

    try:
        configs_dict = get_configs(file_dir=config_file)
        configs = argparse.Namespace(**configs_dict)
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    # 设置路径
    configs.log_dir = os.path.join(ROOT_DIR, "logs", "commnet")
    configs.model_dir = os.path.join(ROOT_DIR, "models", "commnet")
    os.makedirs(configs.log_dir, exist_ok=True)
    os.makedirs(configs.model_dir, exist_ok=True)

    try:
        logger = Logger(configs.log_dir, logger_type=configs.logger)
        logger.save_config(configs)
        print("✓ 日志器初始化成功")
    except Exception as e:
        print(f"❌ 日志器初始化失败: {e}")
        sys.exit(1)

    print("🚀 开始CommNet训练...")

    # 初始化最佳模型保存器
    best_saver = BestModelSaver(configs.model_dir, "commnet")

    try:
        run_commnet(configs, best_saver=best_saver)
        print("✓ CommNet训练完成")
    except Exception as e:
        print(f"❌ CommNet训练失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.close()

        if best_saver.best_model_path:
            print(f"🏆 最佳模型: {best_saver.best_model_path}")
            print(f"📊 最佳奖励: {best_saver.best_reward:.4f}")