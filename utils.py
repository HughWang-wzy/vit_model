import torch
from torchsummary import summary as torch_summary
from thop import profile

def calculate_class_weights(targets):
    """
    根据数据集中各类别样本数量计算类别权重。
    权重与类别样本数成反比。

    Args:
        targets (list or torch.Tensor): 数据集的所有标签。

    Returns:
        torch.Tensor: 每个类别的权重。
    """
    targets = torch.tensor(targets)
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    
    # weight = total_samples / (num_classes * class_counts)
    # 为了避免除以零，给class_counts加上一个很小的数
    weights = total_samples / (len(class_counts) * class_counts.float() + 1e-6)
    
    return weights

def analyze_model_complexity(model, input_size=(3, 32, 32)):
    """
    使用torchsummary和thop分析模型的参数量和FLOPs。

    Args:
        model (nn.Module): 需要分析的模型。
        input_size (tuple): 模型输入的尺寸。
    """
    device = next(model.parameters()).device
    
    # --- 使用 torchsummary 计算参数量 ---
    print("--- Model Summary (Parameters) ---")
    try:
        torch_summary(model, input_size)
    except Exception as e:
        print(f"torchsummary failed: {e}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {params / 1e6:.2f} M")
    
    # --- 使用 thop 计算 FLOPs ---
    print("\n--- Model Complexity (FLOPs) ---")
    dummy_input = torch.randn(1, *input_size).to(device)
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Total parameters: {params / 1e6:.2f} M")
        print(f"Total FLOPs (Multiply-Adds): {flops / 1e9:.2f} G")
    except Exception as e:
        print(f"thop failed to calculate FLOPs: {e}")
    
