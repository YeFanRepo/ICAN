# baselines/commnet/commnet.py（改造后）
from xuance.environment import REGISTRY_ENV
from xuance.environment import make_envs
from xuance.torch.agents import CommNet_Agents


def run_commnet(configs):
    """CommNet训练核心函数，接收统一配置参数"""
    # 如需注册自定义环境，保留此注释（当前用默认环境，无需修改）
    # REGISTRY_ENV[configs.env_name] = MyNewEnv

    envs = make_envs(configs)  # 创建并行环境
    agent = CommNet_Agents(config=configs, envs=envs)  # 初始化智能体（变量名小写，符合PEP8）
    agent.train(configs.running_steps // configs.parallels)  # 启动训练
    agent.save_model("final_train_model.pth")  # 保存最终模型
    agent.finish()  # 释放资源，结束训练
