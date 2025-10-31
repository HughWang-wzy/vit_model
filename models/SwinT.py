import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from typing import Tuple, List, Optional
from itertools import repeat
import collections.abc
import math

# --- 工具函数 ---
def _ntuple(n):
    # 将单个值或可迭代对象转换为 n 元组
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2) # 常用于处理 kernel_size, stride, padding 等

# --- 1. DropPath (Stochastic Depth) ---
# Swin Transformer 标准的正则化技术，随机“跳过”整个 Block
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# --- 2. 核心模块: Window Attention (W-MSA / SW-MSA 的基础) ---
class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 # 注意力缩放因子

        # --- 相对位置偏置 (Relative Position Bias) ---
        # 创建一个可学习的偏置表，大小为 (可能的最大相对距离_h * 可能的最大相对距离_w, 头数)
        # 例如 7x7 窗口，相对距离从 -6 到 +6，共 13 种可能
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 计算窗口内所有 token 对的相对位置索引，并注册为 buffer (不需要梯度)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1 # 行相对坐标平移到 [0, 2*Wh-2]
        relative_coords[:, :, 1] += self.window_size[1] - 1 # 列相对坐标平移到 [0, 2*Ww-2]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 展平成一维索引
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        # --- 相对位置偏置结束 ---

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Q, K, V 线性映射
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # 输出线性映射
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02) # 初始化偏置表
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B*num_windows, N, C), N = window_size[0] * window_size[1]
        B_, N, C = x.shape
        # 计算 Q, K, V: (3, B*nW, nH, N, C/nH)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # 计算注意力分数: (B*nW, nH, N, N)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # --- 加入相对位置偏置 ---
        # 从偏置表中根据索引查找对应的偏置值
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1) # (N, N, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # (nH, N, N)
        attn = attn + relative_position_bias.unsqueeze(0) # (B*nW, nH, N, N)
        # --- 相对位置偏置结束 ---

        # --- 应用 SW-MSA 的掩码 (如果提供了) ---
        if mask is not None:
            nW = mask.shape[0] # mask: (nW, N, N)
            # 将 attn 和 mask 扩展到匹配的维度再相加
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) # 合并回 (B*nW, nH, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        # --- 掩码结束 ---

        attn = self.attn_drop(attn)

        # 计算输出: (B*nW, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- 3. MLP (Feed-Forward Network) ---
# 标准的 Transformer MLP 结构，包含两次线性层和激活函数
class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(in_features * 4) # 默认使用 4 倍扩展
        self.fc1 = nn.Linear(in_features, hidden_features) # 扩展
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features) # 收缩
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --- 4. Swin Transformer Block (核心块) ---
# 包含 W-MSA/SW-MSA 和 MLP
def window_partition(x: Tensor, window_size: int) -> Tensor:
    # 将 (B, H, W, C) 的特征图分割成 (B*nW, Ws, Ws, C) 的窗口
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    # 将 (B*nW, Ws, Ws, C) 的窗口合并回 (B, H, W, C) 的特征图
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 7,
                 shift_size: int = 0, # shift_size > 0 表示使用 SW-MSA
                 mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        ws = to_2tuple(self.window_size)

        # 如果窗口太大，则不进行移位
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        ws = to_2tuple(self.window_size)

        self.norm1 = norm_layer(dim) # LayerNorm (Pre-Norm)
        self.attn = WindowAttention( # W-MSA 或 SW-MSA
            dim, window_size=ws, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim) # LayerNorm (Pre-Norm)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # --- 计算 SW-MSA 的注意力掩码 (如果需要) ---
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1)) # 用于标记窗口区域
            # 定义切片，用于标记不同区域
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt # 给不同区域编号
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size) # (nW, Ws, Ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # (nW, N)
            # 计算掩码：如果两个 token 来自不同的原始区域 (编号不同)，则它们之间的注意力应被屏蔽
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (nW, N, N)
            # 将非 0 (不同区域) 的位置填充为 -100 (softmax 后接近 0)，0 (同区域) 的位置保持 0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None # W-MSA 不需要掩码
        # --- 掩码计算结束 ---

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size L={L}, H={H}, W={W}"

        shortcut = x # 保存残差连接的输入
        x = self.norm1(x) # Pre-Norm
        x = x.view(B, H, W, C)

        # --- 循环位移 (如果 shift_size > 0) ---
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # --- 位移结束 ---

        # --- 窗口划分 ---
        x_windows = window_partition(shifted_x, self.window_size) # (B*nW, Ws, Ws, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) # (B*nW, N, C)
        # --- 划分结束 ---

        # --- W-MSA / SW-MSA (应用掩码) ---
        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (B*nW, N, C)
        # --- 注意力结束 ---

        # --- 窗口合并 ---
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # (B*nW, Ws, Ws, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (B, H, W, C)
        # --- 合并结束 ---

        # --- 逆向循环位移 (如果 shift_size > 0) ---
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # --- 逆向位移结束 ---
        x = x.view(B, H * W, C) # (B, L, C)

        # --- 第一个残差连接 ---
        x = shortcut + self.drop_path(x)

        # --- MLP 部分 ---
        shortcut_mlp = x
        x = self.norm2(x) # Pre-Norm
        x = self.mlp(x)
        x = shortcut_mlp + self.drop_path(x) # 第二个残差连接
        # --- MLP 结束 ---

        return x

# --- 5. Patch 合并 (PatchMerging - 实现层级结构的关键) ---
class PatchMerging(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], dim: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution # (H, W)
        self.dim = dim
        # 将 4 个 Patch 合并，维度变为 4*dim，然后通过线性层降维到 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim) # 在降维前进行归一化

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, C), L = H * W
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"input feature size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # --- 提取 2x2 邻域的 4 个 Patch ---
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C) - 左上
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C) - 左下
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C) - 右上
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C) - 右下
        # --- 提取结束 ---

        # --- 合并与降维 ---
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C) - 在通道维度拼接
        x = x.view(B, -1, 4 * C)  # (B, L/4, 4*C) - 展平空间维度
        x = self.norm(x)
        x = self.reduction(x) # (B, L/4, 2*C) - 维度减半
        # --- 合并结束 ---

        # 输出序列长度减为 1/4，维度加倍
        return x

# --- 6. Swin Transformer Stage (BasicLayer) ---
# 一个 Stage 包含多个 SwinTransformerBlock 和一个可选的 PatchMerging 层
class BasicLayer(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], depth: int, num_heads: int,
                 window_size: int, mlp_ratio: float = 4., qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm, downsample: Optional[nn.Module] = None, # downsample 通常是 PatchMerging
                 use_checkpoint: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint # 是否使用梯度检查点节省显存

        # --- 构建 depth 个 SwinTransformerBlock ---
        # 注意: drop_path 是一个列表，实现逐层递增的 Stochastic Depth
        # shift_size 在奇数层和偶数层之间交替 (0 和 window_size // 2)，实现 W-MSA 和 SW-MSA 的交替
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, # 交替移位
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        # --- Block 构建结束 ---

        # --- Patch Merging 层 (如果不是最后一个 Stage) ---
        if downsample is not None:
            # downsample 参数传入的是 PatchMerging 类
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        # --- Merging 结束 ---

    def forward(self, x: Tensor) -> Tensor:
        # 依次通过所有 Block
        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x) # 使用梯度检查点
            else:
                x = blk(x)
        # 如果有 Downsample 层，则执行
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# --- 7. Patch Embedding Layer (图像到 Patch 序列) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # 计算输出的特征图分辨率 (H', W')
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1] # 总 Patch 数 L = H' * W'

        self.in_chans = in_chans
        self.embed_dim = embed_dim # 输出维度 C

        # 使用一个 stride=patch_size 的卷积层实现 Patch 切分和线性嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Swin 的 PatchEmbed 通常后面会跟一个 Norm 层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 卷积: (B, C, H, W) -> (B, embed_dim, H', W')
        # 展平 + 转置: -> (B, embed_dim, L) -> (B, L, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x) # (B, L, C)
        return x

# --- 8. 最终模型: SwinTransformer (通用模板) ---
class SwinTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: int = 96, # 初始维度 C
                 depths: List[int] = [2, 2, 6, 2], # 每个 Stage 的 Block 数量
                 num_heads: List[int] = [3, 6, 12, 24], # 每个 Stage 的头数
                 window_size: int = 7, # 窗口大小
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop_rate: float = 0., attn_drop_rate: float = 0., drop_path_rate: float = 0.1, # DropPath 速率
                 norm_layer: nn.Module = nn.LayerNorm, ape: bool = False, # 是否使用绝对位置编码 (Swin 通常不用)
                 patch_norm: bool = True, # 是否在 PatchEmbed 后加 Norm (Swin V1/V2 都用)
                 use_checkpoint: bool = False, # 是否使用梯度检查点
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths) # Stage 数量
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # 计算最后一个 Stage 的输出维度
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # --- 1. Patch Embedding ---
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution # (H/4, W/4)

        # --- 2. 绝对位置编码 (可选) ---
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate) # 输入序列的 Dropout

        # --- 3. 构建 Stochastic Depth 的 drop rate 列表 ---
        # 逐层递增的 DropPath 概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # --- 4. 构建所有 Stage ---
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 当前 Stage 的维度 (C, 2C, 4C, 8C)
            current_dim = int(embed_dim * 2 ** i_layer)
            # 当前 Stage 的输入分辨率 (H/4, W/4), (H/8, W/8), ...
            current_input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                        patches_resolution[1] // (2 ** i_layer))
            # 动态调整窗口大小，防止窗口大于特征图
            current_window_size = window_size
            if min(current_input_resolution) < window_size:
                current_window_size = min(current_input_resolution)

            layer = BasicLayer(dim=current_dim,
                               input_resolution=current_input_resolution,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=current_window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], # 分配对应的 DropPath rates
                               norm_layer=norm_layer,
                               # 如果不是最后一个 Stage，则添加 PatchMerging 作为 downsample
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        # --- Stage 构建结束 ---

        # --- 5. 最终的 Norm 和 Head ---
        self.norm = norm_layer(self.num_features) # 在 GAP 之前 Norm
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 全局平均池化
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity() # 分类头
        # --- Head 结束 ---

        self.apply(self._init_weights) # 初始化权重
        print(f"Swin Transformer initialized. Output features: {self.num_features}")

    def _init_weights(self, m):
        # 标准的 Swin 权重初始化
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # --- 用于优化器设置：哪些参数不应使用权重衰减 ---
    @torch.jit.ignore
    def no_weight_decay(self):
        # 绝对位置编码通常不加 weight decay
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
         # 相对位置偏置表通常不加 weight decay
        return {'relative_position_bias_table'}
    # --- 优化器设置结束 ---

    def forward_features(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x) # -> (B, L, C)

        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # --- 依次通过所有 Stage ---
        for layer in self.layers:
            x = layer(x) # 每个 Stage 内部处理 + 可能的下采样
        # --- Stage 结束 ---

        x = self.norm(x)  # (B, L', C_final)
        # --- 全局平均池化 ---
        # (B, L', C_final) -> (B, C_final, L') -> (B, C_final, 1) -> (B, C_final)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        # --- 池化结束 ---
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x) # 提取特征
        x = self.head(x) # 分类
        return x

