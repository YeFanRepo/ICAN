# baselines/tarmac/tarmac.py（改造后）
import argparse
from xuance.common import get_configs
from xuance.environment import REGISTRY_MULTI_AGENT_ENV
from xuance.environment import make_envs
from xuance.torch.agents.multi_agent_rl.tarmac_agents import TarMAC_Agents

# 封装TarMAC训练逻辑，供train目录脚本调用
def run_tarmac(configs):
    # REGISTRY_MULTI_AGENT_ENV[configs.env_name] = MyNewEnv  # 如需自定义环境，保留此行
    envs = make_envs(configs)  # 创建并行环境
    agent = TarMAC_Agents(config=configs, envs=envs)  # 初始化TarMAC智能体（变量名小写，符合PEP8规范）
    agent.train(configs.running_steps // configs.parallels)  # 开始训练
    agent.save_model("final_train_model.pth")  # 保存最终模型
    agent.finish()  # 结束训练，释放资源
