import torch
from tqdm import tqdm
import time
import os
import csv
import torch
from tqdm import tqdm
from timm.data import Mixup

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=10, stage_name="", patience=10, train_dir="./", mixup_cutmix=None):
    since = time.time()
    best_acc = 0.0
    best_epoch = 0
    no_improve_epochs = 0  # 记录连续未提升的 epoch 数

    # 用于绘图的历史记录
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # 自动计算数据集大小
    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'test']}

    csv_path = os.path.join(train_dir, f"{stage_name}_training_log.csv")

    # 创建 CSV 文件并写入表头
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "loss", "accuracy"])


    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0  # --- 修复：用于tqdm的累计样本数 ---

            # 使用 tqdm 显示进度条
            with tqdm(total=len(dataloaders[phase]), desc=f"{phase} Epoch {epoch+1}/{num_epochs}") as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 只在训练阶段进行前向传播和梯度计算
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        # --- 新增: 应用 Mixup/CutMix (仅在训练阶段) ---
                        if phase == 'train' and mixup_cutmix is not None:
                            inputs, labels = mixup_cutmix(inputs, labels)
                        # -----------------------------------------------

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels) # CrossEntropyLoss 会自动处理软标签

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计损失和准确率
                    running_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0) # --- 修复 ---

                    # --- 修改: 准确率计算 (支持软标签) ---
                    if phase == 'train' and mixup_cutmix is not None:
                        # 对于软标签, 比较 argmax
                        running_corrects += torch.sum(preds == labels.argmax(dim=1))
                    else:
                        # 对于硬标签, 直接比较
                        running_corrects += torch.sum(preds == labels.data)
                    # --------------------------------------

                    # --- 修复: 更新进度条 (显示累计平均值) ---
                    current_loss = running_loss / total_samples
                    current_acc = running_corrects.double() / total_samples
                    pbar.set_postfix(loss=current_loss, accuracy=100. * current_acc.item())
                    # ---------------------------------------
                    pbar.update(1)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录历史
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
                    best_acc_path = os.path.join(train_dir, "best_accuracy.txt")
                    with open(best_acc_path, mode="w") as f:
                        f.write(f"Best Accuracy: {best_acc:.4f} at Epoch: {best_epoch+1}\n")
                else:
                    no_improve_epochs += 1

        # 检查早停条件
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val Acc: {best_acc:.4f} at epoch {best_epoch+1}")
            break

        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 if phase == 'test':
                    scheduler.step(epoch_loss)
            else:
                scheduler.step()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    return model, history

def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型的函数。
    (此函数无需更改，评估时不应使用Mixup/CutMix)
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
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