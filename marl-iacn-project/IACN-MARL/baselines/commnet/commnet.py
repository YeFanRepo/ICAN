# baselines/commnet/commnet.py
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import CommNet_Agents


def run_commnet(configs, best_saver=None):
    """CommNet训练核心函数，接收统一配置参数"""
    envs = make_envs(configs)
    agent = CommNet_Agents(config=configs, envs=envs)
    agent.train(configs.running_steps // configs.parallels)

    # 保存最终模型
    agent.save_model("final_train_model.pth")

    # 保存最优模型（如果提供了best_saver）
    if best_saver is not None:
        best_reward = getattr(agent, 'best_eval_reward', None)
        if best_reward is not None:
            best_saver.save_if_better(best_reward, agent)

    agent.finish()