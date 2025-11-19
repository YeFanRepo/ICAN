# baselines/mappo/mappo.py
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents


def run_mappo(configs, best_saver=None):
    # 创建环境
    envs = make_envs(configs)

    # 创建agent
    agent = MAPPO_Agents(config=configs, envs=envs)

    # 训练
    agent.train(configs.running_steps // configs.parallels)

    # 保存最终模型
    agent.save_model("final_train_model.pth")

    # 如果有最佳模型保存器，保存最佳模型
    if best_saver is not None:
        # 这里需要获取训练过程中的最佳奖励
        # 假设我们可以从agent或环境中获取
        best_reward = getattr(agent, 'best_eval_reward', None)
        if best_reward is not None:
            best_saver.save_if_better(best_reward, agent)

    agent.finish()