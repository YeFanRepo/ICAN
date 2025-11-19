# baselines/qmix/qmix.py
from xuance.environment import make_envs
from xuance.torch.agents import QMIX_Agents


def run_qmix(configs, best_saver=None):
    envs = make_envs(configs)
    agent = QMIX_Agents(config=configs, envs=envs)
    agent.train(configs.running_steps // configs.parallels)

    # 保存最终模型
    agent.save_model("final_train_model.pth")

    # 保存最优模型（如果提供了best_saver）
    if best_saver is not None:
        best_reward = getattr(agent, 'best_eval_reward', None)
        if best_reward is not None:
            best_saver.save_if_better(best_reward, agent)

    agent.finish()