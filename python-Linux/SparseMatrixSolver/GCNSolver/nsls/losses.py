import torch

from torch import Tensor
from torch_scatter import scatter_sum

# 定义余弦距离损失类
class CosineDistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        eps = 1e-12
        # 计算预测和目标的范数
        preds_norm = scatter_sum(preds.square(), batch_map).sqrt_()
        target_norm = scatter_sum(target.square(), batch_map).sqrt_()
        # 计算点积和余弦相似度
        dot_product = scatter_sum(preds * target, batch_map)
        cosine = dot_product / torch.clamp_min(preds_norm * target_norm, eps)
        # 计算余弦距离
        cosine_distance = 1 - cosine
        return cosine_distance.mean()

# 定义 L1 距离损失类
class L1DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        # 计算绝对差值和 L1 距离
        absolute_difference = torch.abs(preds - target)
        l1_distance = scatter_sum(absolute_difference, batch_map)
        return l1_distance.mean()

# 定义 L2 距离损失类
class L2DistanceLoss(torch.nn.Module):
    def forward(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> Tensor:
        # 计算平方差值和 L2 距离
        squared_difference = torch.square(preds - target)
        l2_distance = scatter_sum(squared_difference, batch_map).sqrt_()
        return l2_distance.mean()
