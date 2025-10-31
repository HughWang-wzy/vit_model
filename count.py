import os

def count_samples_in_classes(data_dir):
    """
    统计指定路径下每个类别的样本数量。

    Args:
        data_dir (str): 数据集的根目录，每个类别的图片存放在以类别编号命名的子目录中。

    Returns:
        dict: 每个类别及其对应的样本数量。
    """
    class_counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_samples = len(os.listdir(class_path))
            class_counts[class_name] = num_samples
    return class_counts

# CIFAR10_unbalance 路径
data_dir = './CIFAR10_balance'

# 统计样本数量
class_counts = count_samples_in_classes(data_dir)

# 打印结果
for class_name, count in class_counts.items():
    print(f"Class {class_name}: {count} samples")