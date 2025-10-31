import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F # --- [NEW] ---
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
from models.vanilla_vit import ViT # 用于加载教师模型
from models.dynamic_vits import DynamicViT # 我们的学生模型

# --- 风格复制：从 main.py 复制辅助工具 ---
class RandomChoice:
    def __init__(self, transforms, probabilities):
        self.transforms = transforms
        self.probabilities = probabilities
    def __call__(self, inputs, labels):
        transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return transform(inputs, labels)

def get_next_train_dir(base_dir="./dynamic_vit_models"):
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
def train_model(
    student_model, teacher_model, # --- [MODIFIED]: 传入学生和教师
    dataloaders, 
    classify_criterion, distill_criterion, # --- [MODIFIED]: 两个损失函数
    optimizer, scheduler, device, num_epochs=10, 
    stage_name="", patience=10, train_dir="./", mixup_cutmix=None,
    kd_alpha=0.5, kd_temp=4.0 # --- [NEW]: 蒸馏超参数
    ):
    
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

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                student_model.train()  # 学生设为训练模式
                teacher_model.eval()   # 教师始终为评估模式
            else:
                student_model.eval()   # 评估时只评估学生
                teacher_model.eval()

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
                        
                        # --- [MODIFIED]: 知识蒸馏 (KD) 训练逻辑 ---
                        # 1. 获取学生输出
                        student_logits = student_model(inputs)
                        
                        # 2. 计算标准分类损失 (学生 vs 标签)
                        loss_ce = classify_criterion(student_logits, labels)

                        if phase == 'train':
                            # 3. (仅训练时) 获取教师输出
                            with torch.no_grad():
                                teacher_logits = teacher_model(inputs)
                            
                            # 4. (仅训练时) 计算蒸馏损失
                            loss_distill = distill_criterion(
                                F.log_softmax(student_logits / kd_temp, dim=1),
                                F.softmax(teacher_logits / kd_temp, dim=1)
                            )
                            
                            # 5. 组合损失
                            loss = (1.0 - kd_alpha) * loss_ce + (kd_alpha * kd_temp * kd_temp) * loss_distill
                        else:
                            # 评估时，只关心学生自己的分类性能
                            loss = loss_ce
                        
                        # 准确率始终基于学生模型的预测
                        _, preds = torch.max(student_logits, 1)
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
                    # 只保存学生模型
                    torch.save(student_model.state_dict(), os.path.join(train_dir, f"best_model_{stage_name}.pth"))
                    print(f"Saved best STUDENT model for {stage_name} with Acc: {best_acc:.4f}")
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
    
    return student_model, history

# --- 风格复制：从 trainer.py 复制 evaluate_model ---
# (此函数无需修改，因为它只评估传入的 student_model)
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Student Model")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

    # --- [MODIFIED]: 实例化教师和学生 ---
    
    # 1. 实例化教师 (ViT)
    if not args.teacher_path or not os.path.exists(args.teacher_path):
        raise ValueError(f"必须提供有效的 --teacher_path ( {args.teacher_path} ) 才能进行知识蒸馏。")
        
    print(f"正在从 {args.teacher_path} 加载 教师模型 (ViT)...")
    # (确保这些参数与您的教师模型 *完全一致*)
    
    teacher_model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=args.num_classes,
        dim=640,
        depth=10,
        heads=10,
        mlp_dim=2560,
        dropout=0.0,
        emb_dropout=0.05
    ).to(device)

    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher_model.eval() # 设为评估模式
    for param in teacher_model.parameters(): # 冻结教师
        param.requires_grad = False
    print("教师模型加载并冻结完毕。")

    # 2. 实例化学生 (DynamicViT)
    print("正在初始化 学生模型 (DynamicViT)...")
    student_model = DynamicViT(
        image_size=32, patch_size=4, num_classes=args.num_classes,
        dim=640, depth=10, heads=10, mlp_dim=2560, # 必须与教师主干一致
        pruning_loc=[3, 6, 9],
        keep_ratios=[0.75, 0.5, 0.25]
    ).to(device)
    
    # (可选但推荐): 用教师权重初始化学生的主干
    student_model.load_state_dict(teacher_model.state_dict(), strict=False)
    print("学生模型已使用教师权重初始化主干。")
    
    total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"总可训练参数 (学生): {total_params / 1e6:.2f}M")
    
    # --- Mixup 和 损失函数 ---
    mixup_cutmix = RandomChoice(
        transforms=[
            Mixup(num_classes=args.num_classes, mixup_alpha=0.8, cutmix_alpha=0.0),
            Mixup(num_classes=args.num_classes, mixup_alpha=0.0, cutmix_alpha=0.8)
        ],
        probabilities=[0.33, 0.33]
    )
    # [MODIFIED]: 需要两个损失函数
    classify_criterion = nn.CrossEntropyLoss()
    distill_criterion = nn.KLDivLoss(reduction='batchmean')
    
    # [MODIFIED]: 优化器只优化学生模型的参数
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("开始知识蒸馏训练...")
    train_dir = get_next_train_dir(base_dir="./vit_models_v1.4_dynamic_vit")
    current_script = __file__
    shutil.copy(current_script, os.path.join(train_dir, os.path.basename(current_script)))
    print(f"当前脚本已复制到训练文件夹: {train_dir}")

    train_model(
        student_model=student_model,
        teacher_model=teacher_model,
        dataloaders={'train': train_loader, 'test': test_loader},
        classify_criterion=classify_criterion,
        distill_criterion=distill_criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        stage_name="dynamic_vit_kd",
        patience=20,
        train_dir=train_dir,
        mixup_cutmix=mixup_cutmix,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp
    )
    print("训练完成。")

    print("\n开始在测试集上评估 (学生) 模型...")
    best_model_path = os.path.join(train_dir, "best_model_dynamic_vit_kd.pth")
    if os.path.exists(best_model_path):
        student_model.load_state_dict(torch.load(best_model_path, map_location=device))
        evaluate_model(student_model, test_loader, classify_criterion, device)
    else:
        print("未找到最佳模型，跳过最终评估。")
    
    print("评估完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DynamicViT (知识蒸馏) 训练脚本')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    
    # --- [MODIFIED]: 添加教师和蒸馏参数 ---
    parser.add_argument('--teacher_path', type=str, default="vit_models_v1.4/train2/best_model_vit_stage1.pth",
                        help='(必需) 预训练的 vanilla_vit 权重 (.pth) 路径')
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='蒸馏损失的权重 (0.0 到 1.0)')
    parser.add_argument('--kd_temp', type=float, default=4.0,
                        help='知识蒸馏的温度')
    
    args = parser.parse_args()
    main(args)