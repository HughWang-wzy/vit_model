import os
import json

def extract_coatnet_params(main_file_path):
    """
    从 main.py 文件中提取 CoAtNet 模型的参数。
    Args:
        main_file_path (str): main.py 文件的路径。
    Returns:
        dict: 提取的参数字典。
    """
    params = {}
    try:
        with open(main_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("img_size"):
                    params["img_size"] = int(line.split("=")[1].strip().strip(","))
                elif line.startswith("block_types"):
                    block_types_str = line.split("=")[1].strip().strip(",")
                    params["block_types"] = eval(block_types_str)
                elif line.startswith("dims"):
                    dims_str = line.split("=")[1].strip().strip(",")
                    params["dims"] = [int(x) for x in dims_str.strip("[]").split(",")]
                elif line.startswith("depths"):
                    depths_str = line.split("=")[1].strip().strip(",")
                    params["depths"] = [int(x) for x in depths_str.strip("[]").split(",")]
                elif line.startswith("transformer_heads"):
                    params["transformer_heads"] = int(line.split("=")[1].strip().strip(","))
                elif line.startswith("transformer_dim_head"):
                    params["transformer_dim_head"] = int(line.split("=")[1].strip().strip(","))
                elif line.startswith("mbconv_expand_ratios"):
                    ratios_str = line.split("=")[1].strip().strip(",")
                    params["mbconv_expand_ratios"] = [int(x) for x in ratios_str.strip("[]").split(",")]
                elif line.startswith("mbconv_kernel_sizes"):
                    kernel_sizes_str = line.split("=")[1].strip().strip(",")
                    params["mbconv_kernel_sizes"] = [int(x) for x in kernel_sizes_str.strip("[]").split(",")]
                elif line.startswith("drop_path_rate"):
                    params["drop_path_rate"] = float(line.split("=")[1].strip().strip(","))
                elif line.startswith("dropout"):
                    params["dropout"] = float(line.split("=")[1].strip().strip(","))
    except FileNotFoundError:
        print(f"未找到文件: {main_file_path}")
    except Exception as e:
        print(f"解析文件时出错: {e}")
    return params


def extract_best_accuracy(best_acc_file_path):
    """
    从 best_accuracy.txt 文件中提取最佳准确率。
    Args:
        best_acc_file_path (str): best_accuracy.txt 文件的路径。
    Returns:
        tuple: (最佳准确率, 对应的 epoch) 或 None。
    """
    try:
        with open(best_acc_file_path, "r") as f:
            line = f.readline().strip()
            parts = line.split(":")
            accuracy = float(parts[1].split("at")[0].strip())
            epoch = int(parts[2].strip())
            return accuracy, epoch
    except Exception as e:
        print(f"解析文件 {best_acc_file_path} 时出错: {e}")
        return None


def summarize_coatnet_params_multiple(base_dirs, output_file="coatnet_params_summary.json"):
    """
    遍历多个 coatnet_models 目录，提取 CoAtNet 模型参数和最高准确率并保存到 JSON 文件。
    Args:
        base_dirs (list): 包含多个 coatnet_models 根目录的列表。
        output_file (str): 保存参数的 JSON 文件路径。
    """
    summary = {}
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"目录不存在: {base_dir}")
            continue

        print(f"正在处理目录: {base_dir}")
        base_summary = {}
        unique_params = None  # 用于存储去重后的参数
        best_accuracy = {"accuracy": 0.0, "epoch": None, "train_dir": None}  # 用于存储最高准确率

        for train_dir in os.listdir(base_dir):
            train_path = os.path.join(base_dir, train_dir)
            if not os.path.isdir(train_path):
                continue

            # 提取 main.py 中的参数
            main_file_path = os.path.join(train_path, "main.py")
            if os.path.exists(main_file_path):
                params = extract_coatnet_params(main_file_path)
                if params and unique_params is None:
                    unique_params = params  # 只保存一次参数

            # 提取 best_accuracy.txt 中的准确率
            best_acc_file_path = os.path.join(train_path, "best_accuracy.txt")
            if os.path.exists(best_acc_file_path):
                accuracy_data = extract_best_accuracy(best_acc_file_path)
                if accuracy_data:
                    accuracy, epoch = accuracy_data
                    if accuracy > best_accuracy["accuracy"]:
                        best_accuracy = {
                            "accuracy": accuracy,
                            "epoch": epoch,
                            "train_dir": train_dir
                        }

        # 保存去重后的参数
        if unique_params:
            base_summary["params"] = unique_params

        # 保存最高准确率
        if best_accuracy["accuracy"] > 0.0:
            base_summary["best_accuracy"] = best_accuracy

        summary[base_dir] = base_summary

    # 保存到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"参数和最高准确率已保存到 {output_file}")


# 示例调用
base_dirs = [
    "coatnet_models_v0.1",
    "coatnet_models_v0.2",
    "coatnet_models_v0.3",
    "coatnet_models_v0.4"
]

summarize_coatnet_params_multiple(base_dirs)