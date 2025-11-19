# 在 train_qmix.py 中添加
from logger import Logger  # 导入日志类

# 初始化日志器（放在配置加载后）
logger = Logger(configs.log_dir)
logger.save_config(configs)  # 保存超参数

# 训练循环中记录数据
for episode in range(total_episodes):
    # ... 训练逻辑 ...
    logger.add_scalar("reward", episode_reward, episode)  # 记录奖励
    logger.add_scalar("loss", total_loss, episode)        # 记录损失
    logger.write_train_info(f"Episode {episode}: reward={episode_reward:.2f}, loss={total_loss:.4f}")

# 保存最终模型
logger.save_checkpoint(agent, "final_model")

# 训练结束关闭日志器
logger.close()
