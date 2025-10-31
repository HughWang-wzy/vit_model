import torch

# 创建一个形状为 [2, 2, 6] 的数据集
data = torch.arange(2 * 2 * 6).reshape(2, 2, 6)
print("Original data shape:", data.shape)
print("Original data:")
print(data)

# 假设 heads = 3
heads = 3

# 测试 reshape(t.shape[0], -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2)
reshaped_1 = data.reshape(data.shape[0], -1, heads, data.shape[-1] // heads).transpose(1, 2)
print("\nReshaped (1) shape:", reshaped_1.shape)
print("Reshaped (1):")
print(reshaped_1)

# 测试 reshape(t.shape[0], self.heads, -1, t.shape[-1] // self.heads)
reshaped_2 = data.reshape(data.shape[0], heads, -1, data.shape[-1] // heads)
print("\nReshaped (2) shape:", reshaped_2.shape)
print("Reshaped (2):")
print(reshaped_2)