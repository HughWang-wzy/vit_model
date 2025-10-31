import torch
from torch import nn
from torch import Tensor
class ConvStem(nn.Module):
    """
    一个更强大的 Patch Embedding，它使用一个小型 CNN 来注入“归纳偏置”。
    这会提取出更好的局部特征。
    
    它专为 image_size=32, patch_size=4 而设计。
    """
    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        
        # 确保输入符合设计
        assert patch_size == 4, "这个 ConvStem 是为 patch_size=4 专门设计的"
        assert image_size == 32, "这个 ConvStem 是为 image_size=32 专门设计的"
        
        # (N, 3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, dim // 2, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(dim // 2)
        self.relu1 = nn.GELU()
        # (N, dim/2, 16, 16)
        
        self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)
        self.relu2 = nn.GELU()
        # (N, dim, 8, 8)
            
        # CLS 令牌
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 位置编码 (8*8 + 1 = 65)
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size // patch_size)**2 + 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        # 1. 通过“微型 CNN”
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        
        # 2. 展平并调换维度
        # (N, dim, 8, 8) -> (N, dim, 64) -> (N, 64, dim)
        x = x.flatten(2).transpose(1, 2) 
        
        # 3. 添加 CLS Token
        # (N, 64, dim) -> (N, 65, dim)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 4. 添加位置编码
        x = x + self.pos_embed
        
        return x
    
class PatchEmbedding(nn.Module):
    """将图像分割成块并进行线性嵌入"""
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        B = x.shape[0]
        # 卷积 -> 展平 -> 调整维度
        # print("Input shape:", x.shape)
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        # print("Output shape:", x.shape)
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 添加位置编码
        x += self.pos_embedding
        # print("Input shape:", x.shape)
        return x

class Attention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # print("Attention forward called")
        # print("Input shape:", x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=-1) # chunk feature into Q, K, V
        # print("QKV shapes:", [t.shape for t in qkv])
        q, k, v = map(lambda t: t.reshape(t.shape[0], -1 , self.heads, t.shape[-1] // self.heads).transpose(1, 2), qkv)
        # print("Q shape:", q.shape)
        # print("K shape:", k.shape)
        # print("V shape:", v.shape)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        # print("Attention weights shape:", attn.shape)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * (x.shape[-1] // self.heads))
        # print("Attention output shape:", out.shape)
        # exit(0)
        return self.to_out(out)

class FeedForward(nn.Module):
    """MLP模块"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, in_channels=3, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)
        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.patch_embedding(img)
        x = self.dropout(x)
        x = self.transformer(x)
        
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)
