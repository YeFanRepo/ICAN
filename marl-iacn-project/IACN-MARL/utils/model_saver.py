# utils/model_saver.py
import os
import torch
from datetime import datetime


class BestModelSaver:
    """通用的最佳模型保存工具类"""

    def __init__(self, model_dir, algorithm_name=""):
        """
        初始化最佳模型保存器

        Args:
            model_dir: 模型保存目录
            algorithm_name: 算法名称，用于文件名标识
        """
        self.model_dir = model_dir
        self.algorithm_name = algorithm_name
        self.best_reward = float('-inf')
        self.best_model_path = None

    def save_if_better(self, current_reward, agent, step=None):
        """
        如果当前模型更好，则保存

        Args:
            current_reward: 当前模型的评估奖励
            agent: 智能体对象，需要有 save_model 方法
            step: 训练步数（可选）

        Returns:
            bool: 是否保存了新的最佳模型
        """
        if current_reward > self.best_reward:
            self.best_reward = current_reward

            # 删除旧的best model
            if self.best_model_path and os.path.exists(self.best_model_path):
                os.remove(self.best_model_path)

            # 保存新的best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_str = f"_step{step}" if step is not None else ""
            algo_str = f"{self.algorithm_name}_" if self.algorithm_name else ""
            filename = f"best_model_{algo_str}{current_reward:.4f}{step_str}_{timestamp}.pth"
            self.best_model_path = os.path.join(self.model_dir, filename)

            agent.save_model(self.best_model_path)
            print(f"🎉 保存最佳模型: reward = {current_reward:.4f}")
            return True
        return False

    def get_best_info(self):
        """获取最佳模型信息"""
        return {
            'best_reward': self.best_reward,
            'best_model_path': self.best_model_path
        }