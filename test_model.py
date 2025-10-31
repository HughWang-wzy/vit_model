import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import get_cifar10_datasets
from models.vanilla_vit import ViT
from models.SwinT import SwinTransformer
from models.coatnet_cifar import CoAtNet
from trainer import train_model, evaluate_model
from utils import calculate_class_weights, analyze_model_complexity

from torchvision import transforms, models

from models.vanilla_vit import ViT # 用于加载教师模型
from models.dynamic_vits import DynamicViT # 我们的学生模型
from models.dynamic_vits import EarlyExitViT

from timm.data import Mixup
import random
import shutil
import os
class RandomChoice:
    def __init__(self, transforms, probabilities):
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, inputs, labels):
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(inputs, labels)
def get_next_train_dir(base_dir="./vit_models"):
    """
    自动生成下一个可用的训练文件夹名称。
    Args:
        base_dir (str): 基础目录，例如 "./vit_models"。
    Returns:
        str: 下一个可用的训练文件夹路径，例如 "./vit_models/train1"。
    """
    # 确保基础目录存在
    os.makedirs(base_dir, exist_ok=True)

    # 获取所有以 "train" 开头的子文件夹
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("train") and os.path.isdir(os.path.join(base_dir, d))]

    # 提取文件夹编号
    existing_indices = []
    for d in existing_dirs:
        try:
            # 提取文件夹名中的数字部分
            index = int(d.replace("train", ""))
            existing_indices.append(index)
        except ValueError:
            continue

    # 计算下一个编号
    next_index = max(existing_indices, default=0) + 1
    next_dir = os.path.join(base_dir, f"train{next_index}")

    # 创建文件夹
    os.makedirs(next_dir, exist_ok=True)
    return next_dir
def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    model = EarlyExitViT(
        image_size=32, patch_size=4, num_classes=args.num_classes,
        dim=640, depth=10, heads=10, mlp_dim=2560,
        exit_loc=[4, 8] # 示例: 在第4和第8层后退出
    ).to(device)

    # model = DynamicViT(
    #     image_size=32, patch_size=4, num_classes=args.num_classes,
    #     dim=640, depth=10, heads=10, mlp_dim=2560, # 必须与教师主干一致
    #     pruning_loc=[3, 6, 9],
    #     keep_ratios=[0.75, 0.5, 0.25]
    # ).to(device)

    # model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    # # 2b. 替换分类头
    # original_in_features = model.heads.head.in_features
    # model.heads.head = nn.Linear(original_in_features, args.num_classes)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"使用的模型: ViT (CIFAR Config)")
    # print(f"总可训练参数: {total_params / 1e6:.2f}M")
    
    model = model.to(device)

    # 4. 模型复杂度分析
    print("\n模型复杂度分析:")
    analyze_model_complexity(model, (3, 32, 32))
    print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT 轻量化实验')
    parser.add_argument('--model_type', type=str, default='vanilla_vit',
                        choices=['vanilla_vit', 'dynamic_vit', 'hierarchical_vit'],
                        help='选择要运行的模型类型')
    parser.add_argument('--batch_size', type=int, default=128*8, help='批处理大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')

    args = parser.parse_args()
    main(args)
