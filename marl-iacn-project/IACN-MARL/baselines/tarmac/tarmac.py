import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.tarmac_agents import TarMAC_Agents

if __name__ == '__main__':

    configs_dict = get_configs(file_dir="../../configs/smac/tarmac/3m.yaml")
    configs = argparse.Namespace(**configs_dict)
    # REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv

    envs = make_envs(configs)  # Make parallel environments.
    Agent = TarMAC_Agents(config=configs, envs=envs)  # Create a TarMAC agent from XuanCe.
    Agent.train(configs.running_steps // configs.parallels)  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.