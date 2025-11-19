import os
import json
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard 工具


class Logger:
    def __init__(self, log_dir, logger_type="tensorboard"):
        """初始化日志记录器（支持 TensorBoard）"""
        self.log_dir = log_dir
        self.logger_type = logger_type  # 日志类型：tensorboard/wandb/file
        self.scalars = {}  # 存储标量数据 {name: [values]}
        self.train_info = []  # 存储训练信息文本

        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"日志目录已创建: {self.log_dir}")

        # 初始化 TensorBoard 写入器（如果配置为 tensorboard）
        self.tb_writer = None
        if self.logger_type == "tensorboard":
            self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard 日志已启用，路径：{self.log_dir}")

    def add_scalar(self, name, value, step):
        """添加标量数据（同时写入 TensorBoard 和本地文件）"""
        # 写入 TensorBoard（如果启用）
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(name, value, step)

        # 本地保存（兼容原有逻辑，方便后续复盘）
        if name not in self.scalars:
            self.scalars[name] = {'steps': [], 'values': []}
        self.scalars[name]['steps'].append(step)
        self.scalars[name]['values'].append(value)
        np.save(os.path.join(self.log_dir, f"{name}_steps.npy"), self.scalars[name]['steps'])
        np.save(os.path.join(self.log_dir, f"{name}.npy"), self.scalars[name]['values'])

    def save_config(self, config):
        """保存超参数配置（同之前逻辑）"""
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        with open(os.path.join(self.log_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=4)
        # 超参数也写入 TensorBoard（可选，方便对比实验）
        if self.tb_writer is not None:
            for k, v in config_dict.items():
                self.tb_writer.add_text(f"config/{k}", str(v))

    def write_train_info(self, info):
        """记录训练信息文本（同之前逻辑）"""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_line = f"{timestamp} {info}\n"
        self.train_info.append(log_line)
        with open(os.path.join(self.log_dir, "train_info.txt"), 'a') as f:
            f.write(log_line)

    def save_checkpoint(self, model, name="checkpoint"):
        """保存模型检查点（同之前逻辑）"""
        checkpoint_path = os.path.join(self.log_dir, f"{name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        self.write_train_info(f"模型检查点已保存: {checkpoint_path}")

    def close(self):
        """关闭日志器（确保资源释放）"""
        # 确认本地数据保存
        for name in self.scalars:
            np.save(os.path.join(self.log_dir, f"{name}_steps.npy"), self.scalars[name]['steps'])
            np.save(os.path.join(self.log_dir, f"{name}.npy"), self.scalars[name]['values'])
        self.write_train_info("训练结束，日志已保存")

        # 关闭 TensorBoard 写入器
        if self.tb_writer is not None:
            self.tb_writer.close()
            print("TensorBoard 日志已关闭")
        print(f"所有日志已保存至: {self.log_dir}")
