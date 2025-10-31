
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple

from .vanilla_vit import TransformerBlock, PatchEmbedding

# region: --- 1. DynamicViT (动态令牌剪枝) ---
# 对应: Dynamic ViT, 学习阈值 (LTMP)

class TokenPruningModule(nn.Module):
    """ 
    动态令牌剪枝模块 (DynamicViT)
    这是一个可学习的模块，符合 LTMP (学习阈值) 的思想。
    """
    def __init__(self, dim: int):
        super().__init__()
        # 一个简单的线性分类器来预测每个 token 的重要性分数
        # 这是可学习的
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim), # 增加 LayerNorm 提高稳定性
            nn.Linear(dim, 1)
            # nn.Linear(dim, dim // 2),
            # nn.GELU(),
            # nn.Linear(dim // 2, 1)
        )

    def forward(self, x: Tensor, keep_ratio: float) -> Tensor:
        """
        Args:
            x (Tensor): 输入 tokens, shape (B, N_img, C) (不含 [CLS])
            keep_ratio (float): 要保留的 tokens 比例 (例如 0.7)
        Returns:
            Tensor: 保留的 tokens (B, N_kept, C)
        """
        B, N, C = x.shape
        num_keep = int(N * keep_ratio)
        if num_keep == N or N == 0: # 增加 N == 0 的检查
            return x # 不需要剪枝

        # (B, N, C) -> (B, N, 1) -> (B, N)
        scores = self.scorer(x).squeeze(-1)

        # 找到分数最高的 top-k tokens 的索引
        # indices shape: (B, num_keep)
        _, keep_indices = torch.topk(scores, k=num_keep, dim=-1)

        # --- 关键：使用 gather 高效索引 ---
        keep_indices, _ = torch.sort(keep_indices, dim=-1) # 排序
        keep_indices_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, C)
        
        # (B, N, C) -> (B, num_keep, C)
        kept_tokens = torch.gather(x, dim=1, index=keep_indices_expanded)

        return kept_tokens

class DynamicViT(nn.Module):
    """
    集成了动态令牌剪枝的 ViT
    """
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512,
                 depth=12, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                 # DynamicViT 特定参数
                 pruning_loc: List[int] = [3, 6, 9], # 在哪些层后进行剪枝
                 keep_ratios: List[float] = [0.75, 0.5, 0.25] # 对应的保留比例
                 ):
        super().__init__()
        assert len(pruning_loc) == len(keep_ratios), "剪枝位置和保留比例的数量必须匹配"

        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.blocks = nn.ModuleList()
        self.pruners = nn.ModuleList()

        pruner_index = 0
        for i in range(depth):
            self.blocks.append(TransformerBlock(dim, heads, mlp_dim, dropout))
            
            # 如果当前层是剪枝位置
            if (i + 1) in pruning_loc:
                self.pruners.append(TokenPruningModule(dim))
                pruner_index += 1
        
        self.pruning_loc = pruning_loc
        self.keep_ratios = keep_ratios
        self.depth = depth

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        print(f"DynamicViT 初始化: 剪枝位置 {pruning_loc}, 保留比例 {keep_ratios}")

    def forward(self, img):
        x = self.patch_embed(img) # (B, N+1, C)
        x = self.dropout(x)
        
        pruner_index = 0
        for i in range(self.depth):
            x = self.blocks[i](x) # (B, N_current+1, C)
            
            # 检查是否在当前层后进行剪枝
            if (i + 1) in self.pruning_loc:
                # 1. 分离 [CLS] Token 和 图像 Tokens
                cls_token = x[:, :1]        # (B, 1, C)
                img_tokens = x[:, 1:]       # (B, N_current, C)
                
                # 2. 获取当前层对应的剪枝器和比例
                pruner = self.pruners[pruner_index]
                keep_ratio = self.keep_ratios[pruner_index]
                pruner_index += 1
                
                # 3. 执行剪枝
                kept_tokens = pruner(img_tokens, keep_ratio=keep_ratio)
                
                # 4. 重新组合
                x = torch.cat((cls_token, kept_tokens), dim=1)

        # 5. 最终分类
        cls_output = x[:, 0] # 只取 [CLS] Token
        return self.mlp_head(cls_output)

# endregion

# region: --- 2. EarlyExitViT (动态深度 / 提前退出) ---
# 对应: 提前退出 (LGViT)

class EarlyExitHead(nn.Module):
    """
    一个简单的提前退出分类头
    """
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        # LGViT 论文指出浅层用类 CNN 头更好
        # 这里为了简化，我们统一使用标准的 ViT 头
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, N+1, C)
        cls_token = x[:, 0] # (B, C)
        return self.head(cls_token) # (B, num_classes)

class EarlyExitViT(nn.Module):
    """
    集成了动态深度的 ViT (LGViT 思想)
    """
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512,
                 depth=12, heads=8, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                 # EarlyExitViT 特定参数
                 exit_loc: List[int] = [4, 8] # 在第 4 层和第 8 层后添加退出头
                 ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, dim)
        self.dropout = nn.Dropout(emb_dropout)
        
        self.blocks = nn.ModuleList()
        self.exit_heads = nn.ModuleList()
        self.exit_loc = exit_loc
        self.depth = depth
        self.num_classes = num_classes

        for i in range(depth):
            self.blocks.append(TransformerBlock(dim, heads, mlp_dim, dropout))
            
            # 如果当前层是退出位置
            if (i + 1) in exit_loc:
                self.exit_heads.append(EarlyExitHead(dim, num_classes))
        
        # 最终的分类头 (总是在最后一层之后)
        self.final_head = EarlyExitHead(dim, num_classes)
        print(f"EarlyExitViT 初始化: 退出位置 {exit_loc}")

    def forward(self, img: Tensor, 
                exit_threshold: float = -1.0, 
                force_all_heads: bool = False) -> List[Tensor]:
        """
        Args:
            exit_threshold (float): 推理时使用。如果 > 0, 并且*某张图片*的置信度 > 阈值, 则提前退出。
            force_all_heads (bool): 训练时使用。
        """
        
        x = self.patch_embed(img)
        x = self.dropout(x)
        
        B = x.shape[0]
        exit_head_index = 0
        all_logits = [] # 用于训练模式
        
        # 跟踪哪些样本已经退出了 (推理时使用)
        exited_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
        final_logits_output = torch.zeros(B, self.num_classes, device=x.device)

        for i in range(self.depth):
            x = self.blocks[i](x)
            
            if (i + 1) in self.exit_loc:
                head = self.exit_heads[exit_head_index]
                logits = head(x) # (B, num_classes)
                exit_head_index += 1
                
                if force_all_heads:
                    all_logits.append(logits)
                
                # --- 推理时的提前退出逻辑 (per-image, LGViT 思想) ---
                elif exit_threshold > 0 and not self.training:
                    confidence = logits.softmax(dim=-1).max(dim=-1)[0] # (B,)
                    
                    # 找到新退出的样本 (且尚未退出)
                    newly_exited = (confidence > exit_threshold) & (~exited_mask)
                    
                    # 存储它们的 logits
                    final_logits_output[newly_exited] = logits[newly_exited]
                    exited_mask = exited_mask | newly_exited
                    
                    # 如果所有样本都退出了
                    if torch.all(exited_mask):
                        # print(f"All samples exited at layer {i+1}")
                        return [final_logits_output]
                # --- 退出逻辑结束 ---

        # 最终的 Head
        final_logits = self.final_head(x)
        
        if force_all_heads:
            all_logits.append(final_logits)
            return all_logits # 训练时返回所有
        
        # 推理时
        elif exit_threshold > 0 and not self.training:
            # 合并最终 logits 和 之前退出的 logits
            final_logits_output[~exited_mask] = final_logits[~exited_mask]
            return [final_logits_output]
        
        # 默认情况 (推理, 但不启用 early_exit)
        return [final_logits]

# endregion

# region: --- 3. ToMe (令牌合并) - 即插即用辅助函数 ---
# 对应: 令牌合并 (ToMe)

@torch.no_grad()
def merge_tokens_by_similarity(x: Tensor, r: int) -> Tensor:
    """
    基于余弦相似度合并 r 个 token (ToMe 精神的简化实现)
    这是一个 *无参数* 的辅助函数。
    
    Args:
        x (Tensor): 输入 tokens (B, N, C) (不含 [CLS])
        r (int): 要合并/移除的 token 数量
    Returns:
        Tensor: 合并后的 tokens (B, N-r, C)
    """
    B, N, C = x.shape
    if r <= 0:
        return x
    
    num_keep = N - r
    if num_keep == N:
        return x

    # 1. 归一化特征
    x_norm = F.normalize(x, dim=-1)
    
    # 2. 计算相似度矩阵
    sim_matrix = torch.bmm(x_norm, x_norm.transpose(-1, -2))
    diag_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
    sim_matrix.masked_fill_(diag_mask, -1.0) # 移除对角线
    
    # 3. 找到最"冗余"的 r 个 token (最大相似度最高)
    max_sim, _ = sim_matrix.max(dim=-1) # (B, N)
    
    # 4. 找到要保留的 N-r 个 token (最独特的, 即 max_sim 最小的)
    keep_indices = max_sim.topk(k=num_keep, dim=-1, largest=False)[1]
    keep_indices, _ = torch.sort(keep_indices, dim=-1) # (B, num_keep)
    
    # 5. 找到要合并的 r 个 token
    merge_indices = max_sim.topk(k=r, dim=-1, largest=True)[1] # (B, r)
    
    # 6. 收集
    x_keep = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, C))
    x_merge = torch.gather(x, 1, merge_indices.unsqueeze(-1).expand(-1, -1, C))

    # 7. (简化合并策略) 将"合并" token 的信息"分摊"加回到"保留" token
    avg_merged_feature = torch.mean(x_merge, dim=1, keepdim=True)
    # 分摊到保留的 token 上
    x_out = x_keep + (avg_merged_feature / num_keep) 
    
    return x_out

class ToMeBlock(nn.Module):
    """
    这是一个 ToMe *模块*，它包装了原始的 TransformerBlock
    它只在 model.eval() 模式下生效
    """
    def __init__(self, original_block: TransformerBlock, merge_ratio: float = 0.5):
        super().__init__()
        self.original_block = original_block
        self.merge_ratio = merge_ratio

    def forward(self, x: Tensor) -> Tensor:
        # 1. 先通过原始 block
        x = self.original_block(x)
        
        # 2. ToMe 只在推理时 (eval 模式) 激活
        if not self.training:
            cls_token = x[:, :1]
            img_tokens = x[:, 1:]
            
            # 3. 计算要合并多少
            # 确保 img_tokens 不为空
            if img_tokens.shape[1] > 0:
                r = int(img_tokens.shape[1] * self.merge_ratio)
                
                # 4. 执行合并
                merged_tokens = merge_tokens_by_similarity(img_tokens, r)
                
                # 5. 重新组合
                x = torch.cat((cls_token, merged_tokens), dim=1)
        
        return x

def apply_tome_to_model(model: nn.Module, 
                        merge_loc: List[int], 
                        merge_ratios: List[float]) -> nn.Module:
    """
    这是一个"即插即用"函数。
    它会*修改*一个已经训练好的 ViT 模型, 将 ToMeBlock 包装进去。
    
    Args:
        model: 你已经训练好的 ViT 模型 (e.g., StandardViT 或 DynamicViT)
        merge_loc: 要插入 ToMe 的层
        merge_ratios: 对应的合并比例
    Returns:
        修改后的模型
    """
    print("正在为模型 (即插即用) 应用 ToMe...")
    merger_index = 0
    
    # 假设你的 ViT 模型有一个 'blocks' 属性 (nn.ModuleList)
    if not hasattr(model, 'blocks') or not isinstance(model.blocks, nn.ModuleList):
        print("错误: ToMe 无法应用, 找不到 'model.blocks' (nn.ModuleList)")
        return model
        
    if len(merge_loc) != len(merge_ratios):
        raise ValueError("merge_loc 和 merge_ratios 的长度必须匹配")

    for i in range(len(model.blocks)):
        if (i + 1) in merge_loc:
            ratio = merge_ratios[merger_index]
            print(f"  在第 {i+1} 层后应用 ToMe, 合并比例 {ratio}")
            
            # 动态替换
            model.blocks[i] = ToMeBlock(
                original_block=model.blocks[i], 
                merge_ratio=ratio
            )
            merger_index += 1
            if merger_index >= len(merge_ratios):
                break # 已经应用了所有指定的 ToMe 块
            
    return model