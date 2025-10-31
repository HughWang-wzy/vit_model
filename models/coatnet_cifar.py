import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple
from torch import Tensor
from timm.models.layers import DropPath
# --- 辅助模块 ---
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

class Attention(nn.Module):
    """ Standard Multi-Head Self-Attention with Relative Position Bias """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rel_pos_bias=False, input_size=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.rel_pos_bias = rel_pos_bias

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        # Relative Position Bias
        if self.rel_pos_bias and input_size:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * input_size[0] - 1) * (2 * input_size[1] - 1), heads))
            coords_h = torch.arange(input_size[0])
            coords_w = torch.arange(input_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1) # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += input_size[0] - 1
            relative_coords[:, :, 1] += input_size[1] - 1
            relative_coords[:, :, 0] *= 2 * input_size[1] - 1
            relative_position_index = relative_coords.sum(-1) # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None


    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Add Relative Position Bias
        if self.rel_pos_bias:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.relative_position_index.shape[0], self.relative_position_index.shape[1], -1) # N, N, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # nH, N, N
            dots = dots + relative_position_bias.unsqueeze(0)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    """ Transformer Block """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rel_pos_bias=False, input_size=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rel_pos_bias=rel_pos_bias, input_size=input_size)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# --- MBConv (来自 EfficientNet / MobileNetV3) ---
class SqueezeExcitation(nn.Module):
    """ Squeeze-and-Excitation Module """
    def __init__(self, input_channels: int, squeeze_ratio: float = 0.25):
        super().__init__()
        squeeze_channels = max(1, int(input_channels * squeeze_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.act = nn.SiLU(inplace=True) # SiLU (Swish) is common in EfficientNet variants
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        scale = self.gate(scale)
        return x * scale

class MBConv(nn.Module):
    """ Mobile Inverted Residual Bottleneck Block """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 expand_ratio: float, se_ratio: float = 0.25, drop_path: float = 0.,
                 norm_layer: nn.Module = nn.BatchNorm2d, act_layer: nn.Module = nn.SiLU):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        use_res_connect = stride == 1 and in_channels == out_channels

        layers: List[nn.Module] = []
        # Expand
        if expand_ratio != 1:
            layers.extend([
                norm_layer(in_channels),
                act_layer(inplace=True),
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            ])
        # Depthwise
        layers.extend([
            norm_layer(hidden_dim),
            act_layer(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
        ])
        # Squeeze-Excitation
        if se_ratio > 0:
            layers.append(SqueezeExcitation(hidden_dim, squeeze_ratio=se_ratio))
        # Project
        layers.extend([
            norm_layer(hidden_dim),
            act_layer(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
        ])

        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_res_connect = use_res_connect

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.drop_path(self.block(x))
        else:
            return self.drop_path(self.block(x))

# --- CoAtNet 主模型 ---
class CoAtNet(nn.Module):
    def __init__(self, img_size=32, in_channels=3, num_classes=10,
                 # 定义每个 Stage 的 Block 类型 ('C' for MBConv, 'T' for Transformer)
                 block_types=['C', 'C', 'T', 'T'],
                 # 定义每个 Stage 的输出通道数
                 dims=[64, 128, 256, 512],
                 # 定义每个 Stage 的 Block 数量
                 depths=[2, 2, 6, 2],
                 # Transformer 相关参数 (仅对 'T' Stage 有效)
                 transformer_heads=8, transformer_dim_head=32, transformer_mlp_dim=None, rel_pos_bias=True,
                 # MBConv 相关参数 (仅对 'C' Stage 有效)
                 mbconv_expand_ratios=[4, 4], # Expand ratio for C stages
                 mbconv_kernel_sizes=[3, 3], # Kernel size for C stages
                 se_ratio=0.25, drop_path_rate=0.1, dropout=0.1):
        super().__init__()
        assert len(block_types) == len(dims) == len(depths)
        num_stages = len(block_types)

        # --- Stem (初始卷积层) ---
        # 使用两层卷积进行下采样，类似 ConvStem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1, bias=False), # Changed stride to 1
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(inplace=True)
        )
        # CIFAR 32x32 -> 16x16 (after stride 2) -> 16x16 (after stride 1)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        drop_path_index = 0

        current_stride = 2 # Initial stride from stem
        current_resolution = (img_size // current_stride, img_size // current_stride) # Start with 16x16

        # --- 构建 Stages ---
        self.stages = nn.ModuleList()
        in_dim = dims[0]

        for i in range(num_stages):
            stage_type = block_types[i]
            out_dim = dims[i]
            depth = depths[i]
            
            # Determine stride for the first block of the stage (except the very first stage after stem)
            # We want resolution H/4, H/8, H/16, H/32 for stages 0, 1, 2, 3 output
            stride = 2 if i > 0 else 1 # Downsample between stages (except after stem)

            if stage_type == 'C':
                # --- MBConv Stage ---
                blocks = []
                # Find MBConv specific params for this stage (assuming len matches num 'C' stages)
                expand_ratio = mbconv_expand_ratios[i if i < len(mbconv_expand_ratios) else -1]
                kernel_size = mbconv_kernel_sizes[i if i < len(mbconv_kernel_sizes) else -1]

                for j in range(depth):
                    block_stride = stride if j == 0 else 1 # Only first block downsamples
                    blocks.append(MBConv(in_dim, out_dim, kernel_size, block_stride,
                                         expand_ratio, se_ratio, dpr[drop_path_index],
                                         norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU))
                    in_dim = out_dim # Update in_dim for next block
                    drop_path_index += 1
                self.stages.append(nn.Sequential(*blocks))

            elif stage_type == 'T':
                # --- Transformer Stage ---
                # First, apply downsampling if needed (stride=2) using a Conv layer
                # Transformer expects same input/output channels per stage
                if stride == 2:
                    downsample_layer = nn.Sequential(
                         nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                         nn.BatchNorm2d(out_dim)
                    )
                    self.stages.append(downsample_layer)
                    in_dim = out_dim # Update in_dim
                    current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)

                # Then, apply Transformer blocks
                # Ensure mlp_dim is calculated correctly (4*dim)
                t_mlp_dim = transformer_mlp_dim if transformer_mlp_dim else out_dim * 4
                
                # Input size for relative position bias needs careful calculation
                transformer_input_size = current_resolution # Use the resolution before this stage's blocks
                
                self.stages.append(
                    Transformer(out_dim, depth, transformer_heads, transformer_dim_head,
                                t_mlp_dim, dropout, rel_pos_bias=rel_pos_bias, input_size=transformer_input_size)
                )
                drop_path_index += depth # Transformer block doesn't use DropPath in this setup
                in_dim = out_dim # Dimension remains the same through Transformer stage

            # Update resolution if downsampling occurred in this stage
            if stride == 2:
                 current_stride *= 2
                 # We already updated current_resolution if it was a 'T' stage with stride=2
                 # If it was a 'C' stage, update now
                 if stage_type == 'C':
                      current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)


        # --- Head ---
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
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
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
             # Handle Transformer stage needing reshape
            is_transformer_stage = isinstance(stage, Transformer)
            if is_transformer_stage:
                 # Transformer expects (B, N, C)
                 B, C, H, W = x.shape
                 x = x.flatten(2).transpose(1, 2) # (B, H*W, C)
                 x = stage(x)
                 # Reshape back to (B, C, H, W) if next stage is Conv
                 # Find the index of the current stage to look ahead
                 current_stage_index = -1
                 for idx, s in enumerate(self.stages):
                     if s is stage:
                         current_stage_index = idx
                         break
                 
                 is_last_stage = current_stage_index == len(self.stages) -1
                 next_stage_is_conv = False
                 if not is_last_stage:
                     next_stage = self.stages[current_stage_index + 1]
                     # Check if next stage contains MBConv (assuming sequential for C) or is the initial Conv downsample
                     if isinstance(next_stage, nn.Sequential) and any(isinstance(m, MBConv) for m in next_stage):
                          next_stage_is_conv = True
                     elif isinstance(next_stage, nn.Sequential) and isinstance(next_stage[0], nn.Conv2d): # Check for Conv downsample layer
                          next_stage_is_conv = True


                 if next_stage_is_conv or is_last_stage: # Reshape if followed by Conv or if it's the end
                     x = x.transpose(1, 2).reshape(B, C, H, W) # (B, C, H, W)
            else:
                 # Conv stages expect (B, C, H, W)
                 x = stage(x)

        x = self.head(x)
        return x
