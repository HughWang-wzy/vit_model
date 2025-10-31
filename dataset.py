import os
from torchvision import datasets, transforms
from torchvision.datasets import DatasetFolder
from PIL import Image

def get_cifar10_datasets(balance_dir='./CIFAR10_balance', unbalance_dir='./CIFAR10_unbalance'):
    transform_train = transforms.Compose([
        transforms.Resize(32), # ViT-B/16 预训练的尺寸
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), # 相比 padding crop 更好
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载不平衡训练集
    train_dataset = CIFAR10Local(root_dir=unbalance_dir, transform=transform_train)

    # 加载平衡测试集
    test_dataset = CIFAR10Local(root_dir=balance_dir, transform=transform_test)

    return train_dataset, test_dataset


class CIFAR10Local(DatasetFolder):
    def __init__(self, root_dir, transform=None):
        super().__init__(root=root_dir, loader=self.pil_loader, extensions=('png', 'jpg', 'jpeg'))
        self.transform = transform

    def pil_loader(self, path):
        """
        加载图片并转换为 RGB 格式。
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        """
        获取指定索引的数据和标签。
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target