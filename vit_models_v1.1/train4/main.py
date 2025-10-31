import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import get_cifar10_datasets
from models.vanilla_vit import ViT
from models.dynamic_vits import DynamicViT, HierarchicalViT
from trainer import train_model, evaluate_model
from utils import calculate_class_weights, analyze_model_complexity

from torchvision import transforms, models

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
        
    """
    主函数，用于执行整个实验流程。
    """
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    # 2. 加载数据集
    print("开始加载CIFAR-10数据集...")
    train_dataset, test_dataset = get_cifar10_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("数据集加载完成。")

    # 3. 初始化模型
    print(f"正在初始化模型: {args.model_type}")
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=args.num_classes,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    # 2a. 加载预训练模型
    # model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    # 2b. 替换分类头
    # original_in_features = model.heads.head.in_features
    # model.heads.head = nn.Linear(original_in_features, args.num_classes)

    # model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1).to(device)
    # # 2b. 替换分类头
    # original_in_features = model.head.in_features
    # model.head = nn.Linear(original_in_features, args.num_classes)
    
    model = model.to(device)

    # 4. 模型复杂度分析
    # print("\n模型复杂度分析:")
    # analyze_model_complexity(model, (3, 224, 224))
    # print("-" * 50)

    # 5. 定义损失函数和优化器
    # --- 类不平衡问题解决方案 ---
    print("正在计算类别权重以解决数据不平衡问题...")
    # 注意：此处需要访问dataset的内部targets，实际应用中请确保dataset类支持此操作

    mixup_cutmix = RandomChoice(
        transforms=[
            Mixup(num_classes=args.num_classes, mixup_alpha=0.6, cutmix_alpha=0.0),
            Mixup(num_classes=args.num_classes, mixup_alpha=0.0, cutmix_alpha=0.8)
        ],
        probabilities=[0.5, 0.5]
    )
    
    print("已启用 MixUp 和 CutMix。")

    # --- 关键步骤 2: 修改损失函数 ---
    # Mixup/CutMix 
    criterion = nn.CrossEntropyLoss() 

    print("类别权重计算完成，已应用到损失函数中。")
    # --------------------------

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 6. 训练和评估模型
    print("开始训练模型...")

    train_dir = get_next_train_dir(base_dir="./vit_models_v1.1")
    # print(f"训练文件夹已设置为: {train_dir}")
    current_script = __file__  # 当前脚本文件路径
    shutil.copy(current_script, os.path.join(train_dir, "main.py"))
    print(f"当前脚本已复制到训练文件夹: {train_dir}")

    pretrained_path = "vit_models_v1.1/train3/best_model_vit_stage1.pth"
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print(f"已加载预训练权重: {pretrained_path}")

    # print("替换分类头...")
    # model.heads.head = nn.Sequential(
    #     nn.Linear(original_in_features, 512),
    #     nn.ReLU(), # ViT 标配的激活函数
    #     # nn.Dropout(0.1), 
    #     nn.Linear(512, 2)
    # ).to(device)
    # print("冻结 backbone...")
    # for name, param in model.named_parameters():
    #     if not name.startswith("heads.head"):  # 只训练分类头
    #         param.requires_grad = False
    
    train_model(
        model=model,
        dataloaders={'train': train_loader, 'test': test_loader},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        stage_name="vit_stage1",
        patience=20,
        train_dir=train_dir,
        mixup_cutmix=mixup_cutmix
        # mixup_cutmix=None
    )
    # train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs=args.epochs)
    print("训练完成。")

    print("\n开始在平衡测试集上评估模型...")
    evaluate_model(model, test_loader, criterion, device)
    print("评估完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViT 轻量化实验')
    parser.add_argument('--model_type', type=str, default='vanilla_vit',
                        choices=['vanilla_vit', 'dynamic_vit', 'hierarchical_vit'],
                        help='选择要运行的模型类型')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--lr', type=float, default=3.3e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')

    args = parser.parse_args()
    main(args)
