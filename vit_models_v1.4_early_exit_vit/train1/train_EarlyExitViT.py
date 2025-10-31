import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import time
import csv
from tqdm import tqdm
from timm.data import Mixup
import random

# --- 假设的导入路径 ---
# (请确保这些路径与您的项目结构一致)
from dataset import get_cifar10_datasets
from models.vanilla_vit import ViT # 用于加载权重
from models.dynamic_vits import EarlyExitViT

# --- 风格复制：从 main.py 复制辅助工具 ---
class RandomChoice:
    def __init__(self, transforms, probabilities):
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, inputs, labels):
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(inputs, labels)

def get_next_train_dir(base_dir="./early_exit_vit_models"):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("train") and os.path.isdir(os.path.join(base_dir, d))]
    existing_indices = []
    for d in existing_dirs:
        try:
            index = int(d.replace("train", ""))
            existing_indices.append(index)
        except ValueError:
            continue
    next_index = max(existing_indices, default=0) + 1
    next_dir = os.path.join(base_dir, f"train{next_index}")
    os.makedirs(next_dir, exist_ok=True)
    return next_dir

# --- 风格复制：从 trainer.py 复制并修改 train_model ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10, stage_name="", patience=10, train_dir="./", mixup_cutmix=None):
    since = time.time()
    best_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0 

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'test']}
    csv_path = os.path.join(train_dir, f"{stage_name}_training_log.csv")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "accuracy"])
        
    # --- [MODIFIED]: 定义多头损失的权重 ---
    # 假设有 2 个退出头 + 1 个最终头
    loss_weights = [0.5, 0.7, 1.0] # 越深的头权重越大
    # ------------------------------------

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        
                        if phase == 'train' and mixup_cutmix is not None:
                            inputs, labels = mixup_cutmix(inputs, labels)
                        
                        # --- [MODIFIED]: EarlyExitViT 训练逻辑 ---
                        # 训练和验证时，我们都需要所有头的输出来计算损失
                        all_logits_list = model(inputs, force_all_heads=True)
                        
                        total_loss = 0
                        
                        # 确保权重列表和头输出列表长度一致
                        if len(loss_weights) != len(all_logits_list):
                            print(f"警告: 损失权重数量({len(loss_weights)}) 与 头输出数量({len(all_logits_list)}) 不匹配! 使用等权重。")
                            current_weights = [1.0] * len(all_logits_list)
                        else:
                            current_weights = loss_weights
                            
                        # 为每个头的输出计算加权损失
                        for i, logits in enumerate(all_logits_list):
                            total_loss += current_weights[i] * criterion(logits, labels)
                        
                        loss = total_loss
                        
                        # 准确率只基于 *最后一个头* (最深的头) 的性能
                        _, preds = torch.max(all_logits_list[-1], 1)
                        # -----------------------------------------

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

                    if phase == 'train' and mixup_cutmix is not None:
                        running_corrects += torch.sum(preds == labels.argmax(dim=1))
                    else:
                        running_corrects += torch.sum(preds == labels.data)

                    current_loss = running_loss / total_samples
                    current_acc = running_corrects.double() / total_samples
                    pbar.set_postfix(loss=current_loss, accuracy=100. * current_acc.item())
                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, phase, epoch_loss, epoch_acc.item()])

            if phase == 'test':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    no_improve_epochs = 0
                    torch.save(model.state_dict(), os.path.join(train_dir, f"best_model_{stage_name}.pth"))
                    print(f"Saved best model for {stage_name} with Acc: {best_acc:.4f}")
                else:
                    no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}")
            break

        if scheduler is not None:
            scheduler.step()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    return model, history

# --- 风格复制：从 trainer.py 复制并修改 evaluate_model ---
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating (Inference Mode)")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # --- [MODIFIED]: 评估时, 使用模型的默认推理逻辑 ---
            # (即 force_all_heads=False, 它会根据阈值自动退出)
            # (它返回一个 list, 我们取第一个元素)
            outputs_list = model(inputs, force_all_heads=False)
            outputs = outputs_list[0] # 
            # -----------------------------------------------

            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            progress_bar.set_postfix(loss=running_loss/total_samples, accuracy=100. * correct_predictions/total_samples)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = correct_predictions / len(test_loader.dataset)
    
    print(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

# --- 风格复制：从 main.py 复制 main 函数 ---
def main(args):
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是: {device}")

    print("开始加载CIFAR-10数据集...")
    train_dataset, test_dataset = get_cifar10_datasets()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("数据集加载完成。")

    print(f"正在初始化模型: EarlyExitViT")
    # --- [MODIFIED]: 实例化 EarlyExitViT ---
    # (确保这些参数与您的教师模型的主干一致)
    model = EarlyExitViT(
        image_size=32, patch_size=4, num_classes=args.num_classes,
        dim=640, depth=10, heads=10, mlp_dim=2560,
        exit_loc=[4, 8] # 示例: 在第4和第8层后退出
    ).to(device)

    # --- [MODIFIED]: 加载预训练的 ViT 权重 ---
    if not args.teacher_path:
        print("警告: 未提供 --teacher_path。模型将从头开始训练。")
    else:
        if not os.path.exists(args.teacher_path):
            print(f"警告: 教师路径 {args.teacher_path} 不存在。模型将从头开始训练。")
        else:
            print(f"正在从 {args.teacher_path} 加载预训练权重...")
            pretrained_weights = torch.load(args.teacher_path, map_location=device)
            # strict=False 允许加载匹配的键 (主干), 忽略不匹配的键 (exit_heads)
            model.load_state_dict(pretrained_weights, strict=False)
            print("权重加载完毕。")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params / 1e6:.2f}M")
    
    # --- Mixup 和 损失函数 ---
    mixup_cutmix = RandomChoice(
        transforms=[
            Mixup(num_classes=args.num_classes, mixup_alpha=0.8, cutmix_alpha=0.0),
            Mixup(num_classes=args.num_classes, mixup_alpha=0.0, cutmix_alpha=0.8)
        ],
        probabilities=[0.33, 0.33]
    )
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("开始训练模型...")
    train_dir = get_next_train_dir(base_dir="./vit_models_v1.4_early_exit_vit")
    current_script = __file__
    shutil.copy(current_script, os.path.join(train_dir, os.path.basename(current_script)))
    print(f"当前脚本已复制到训练文件夹: {train_dir}")

    train_model(
        model=model,
        dataloaders={'train': train_loader, 'test': test_loader},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        stage_name="early_exit_vit",
        patience=20,
        train_dir=train_dir,
        # mixup_cutmix=mixup_cutmix
        mixup_cutmix=None
    )
    print("训练完成。")

    print("\n开始在测试集上评估模型 (使用推理模式)...")
    # 重新加载最佳模型进行最终评估
    best_model_path = os.path.join(train_dir, "best_model_early_exit_vit.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluate_model(model, test_loader, criterion, device)
    else:
        print("未找到最佳模型，跳过最终评估。")
    
    print("评估完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EarlyExitViT 训练脚本')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率 (微调时可能需要更小, e.g., 1e-5)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    # --- [MODIFIED]: 添加教师路径参数 ---
    parser.add_argument('--teacher_path', type=str, default="vit_models_v1.4/train2/best_model_vit_stage1.pth",
                        help='(必需) 预训练的 vanilla_vit 权重 (.pth) 路径')
    args = parser.parse_args()
    main(args)