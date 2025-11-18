import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import MAPPO_Agents

if __name__ == '__main__':

    configs_dict = get_configs(file_dir="../../configs/smac/mappo/3m.yaml")
    configs = argparse.Namespace(**configs_dict)
    # REGISTRY_ENV[configs.env_name] = MyNewEnv

    envs = make_envs(configs)
    Agent = MAPPO_Agents(config=configs, envs=envs)
    Agent.train(configs.running_steps // configs.parallels)
    Agent.save_model("final_train_model.pth")
    Agent.finish()  # Finish the training.