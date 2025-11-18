# train/train_qmix.py（训练入口，统一路径）
import os
import sys
import argparse  # 新增：导入argparse模块，解决NameError

# 把项目根目录加入Python路径（确保能导入baselines和utils）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.qmix.qmix import run_qmix  # 导入baselines的算法核心
from xuance.common import get_configs

if __name__ == '__main__':
    # 配置文件路径：基于项目根目录（train目录的上级是根目录，所以用 ../configs/...）
    config_path = "../configs/smac/qmix/3m.yaml"
    # 读取配置（统一用绝对路径，避免运行目录影响）
    configs_dict = get_configs(file_dir=os.path.abspath(config_path))
    configs = argparse.Namespace(**configs_dict)
    # baselines/qmix/qmix.py（改造后）
    import argparse
    from xuance.common import get_configs
    from xuance.environment import make_envs
    from xuance.torch.agents import QMIX_Agents


    # 删掉原有的 if __name__ == '__main__': 入口代码
    # 新增：封装训练逻辑为函数，供 train 目录脚本调用
    def run_qmix(configs):
        envs = make_envs(configs)
        agent = QMIX_Agents(config=configs, envs=envs)
        agent.train(configs.running_steps // configs.parallels)
        agent.save_model("final_train_model.pth")
        agent.finish()


    # 关键：强制统一日志/模型保存路径（覆盖配置文件中的相对路径，避免混乱）
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configs.log_dir = os.path.join(root_dir, "logs/qmix/")
    configs.model_dir = os.path.join(root_dir, "models/qmix/")

    # 调用baselines的核心训练函数
    run_qmix(configs)
