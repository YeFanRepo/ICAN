import os


def list_directory_contents(path, description):
    """列出指定目录的内容，并处理路径不存在的情况"""
    # 转换为绝对路径，方便查看实际位置
    abs_path = os.path.abspath(path)
    print(f"\n===== 正在查看 {description} 目录 =====")
    print(f"目录绝对路径：{abs_path}")

    try:
        # 检查路径是否存在
        if not os.path.exists(abs_path):
            print(f"❌ 错误：{description} 目录不存在！")
            return

        # 检查是否是目录
        if not os.path.isdir(abs_path):
            print(f"❌ 错误：{abs_path} 不是一个目录！")
            return

        # 列出目录中的文件和子目录
        contents = os.listdir(abs_path)
        if not contents:
            print(f"⚠️ 警告：{description} 目录为空！")
            return

        # 分类显示文件和子目录
        print(f"✅ 找到 {len(contents)} 个内容：")
        print("  子目录：")
        for item in contents:
            item_path = os.path.join(abs_path, item)
            if os.path.isdir(item_path):
                print(f"    - {item}/")
        print("  文件：")
        for item in contents:
            item_path = os.path.join(abs_path, item)
            if os.path.isfile(item_path):
                print(f"    - {item}")

    except Exception as e:
        print(f"❌ 读取目录时出错：{str(e)}")


if __name__ == "__main__":
    # 根据你的项目结构，配置 logs 和 models 的路径
    # 假设 test.py 在 IACN-MARL/ 目录下（与 logs、models 同级）
    # 如果 test.py 位置不同，需调整相对路径（例如："../logs/qmix/" 等）
    logs_path = "./logs/qmix/"  # QMIX 日志目录
    models_path = "./models/qmix/"  # QMIX 模型目录

    # 读取并显示日志目录内容
    list_directory_contents(logs_path, "QMIX 日志")

    # 读取并显示模型目录内容
    list_directory_contents(models_path, "QMIX 模型")

    # 额外提示：检查路径是否与配置文件一致
    print("\n===== 路径检查提示 =====")
    print("1. 若上述目录不存在，请检查配置文件中的 log_dir 和 model_dir 是否正确")
    print("2. 若路径错误，需修改 3m.yaml 中的 log_dir 和 model_dir 为正确相对路径")
    print("3. 确保训练脚本运行时的当前目录与路径匹配（例如在 baselines/qmix/ 目录运行）")
