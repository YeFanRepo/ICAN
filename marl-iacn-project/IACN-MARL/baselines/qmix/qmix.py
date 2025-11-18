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
