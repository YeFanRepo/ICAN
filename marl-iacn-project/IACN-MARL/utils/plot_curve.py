import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def smooth_curve(y, window_size=10):
    """平滑曲线（减少波动）"""
    if len(y) <= window_size:
        return y
    y_smooth = np.convolve(y, np.ones(window_size) / window_size, mode='same')
    return y_smooth[:len(y)]  # 保证长度一致


def plot_curve(log_dir):
    """绘制日志中的曲线"""
    # 检查日志目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误: 日志目录不存在 - {log_dir}")
        return

    # 获取所有标量数据文件
    scalar_files = [f for f in os.listdir(log_dir) if f.endswith('.npy') and not f.endswith('_steps.npy')]
    if not scalar_files:
        print(f"警告: 未在 {log_dir} 找到标量数据文件")
        return

    # 绘制每个标量的曲线
    for file in scalar_files:
        name = file[:-4]  # 移除.npy后缀
        steps_file = f"{name}_steps.npy"

        # 检查步骤文件是否存在
        if not os.path.exists(os.path.join(log_dir, steps_file)):
            print(f"警告: 未找到 {steps_file}，跳过 {file}")
            continue

        # 加载数据
        values = np.load(os.path.join(log_dir, file))
        steps = np.load(os.path.join(log_dir, steps_file))

        # 创建图形
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # x轴只显示整数

        # 绘制原始曲线和平滑曲线
        plt.plot(steps, values, alpha=0.3, label='原始数据')
        plt.plot(steps, smooth_curve(values), label='平滑曲线')

        # 设置图表属性
        plt.title(f'{name.capitalize()} Curve', fontsize=14)
        plt.xlabel('Steps', fontsize=12)
        plt.ylabel(name.capitalize(), fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()

        # 保存图片
        save_path = os.path.join(log_dir, f'{name}_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存曲线: {save_path}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='绘制训练曲线')
    parser.add_argument('--logdir', type=str, required=True, help='日志目录路径')
    args = parser.parse_args()

    # 绘制曲线
    plot_curve(args.logdir)
