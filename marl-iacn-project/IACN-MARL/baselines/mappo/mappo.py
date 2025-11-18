# baselines/mappo/mappo.py（改造后）
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents

# 封装MAPPO训练逻辑为函数，供train目录调用
def run_mappo(configs):
    # REGISTRY_ENV[configs.env_name] = MyNewEnv  # 如需自定义环境，在这里添加
    envs = make_envs(configs)
    agent = MAPPO_Agents(config=configs, envs=envs)  # 注意：变量名小写（PEP8规范）
    agent.train(configs.running_steps // configs.parallels)
    agent.save_model("final_train_model.pth")
    agent.finish()
