from typing import Optional

from torch import Tensor
import torch.nn as nn
import torch_geometric.nn

# 图卷积块
class GraphBlock(nn.Module):
    def __init__(self, width: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置图卷积块的宽度
        self.width = width
        # 实例化图归一化层
        self.graph_norm = torch_geometric.nn.GraphNorm(width)
        # 实例化第一个图卷积层
        self.graph_conv1 = torch_geometric.nn.conv.GraphConv(width, width).jittable()
        # 实例化激活函数
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        # 实例化第二个图卷积层
        self.graph_conv2 = torch_geometric.nn.conv.GraphConv(width, width).jittable()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        # 对输入进行图归一化
        xx = self.graph_norm(x, batch_map)
        # 通过第一个图卷积层进行前向传播
        xx = self.graph_conv1(xx, edge_index, edge_weight)
        # 应用激活函数
        xx = self.activation(xx)
        # 通过第二个图卷积层进行前向传播
        xx = self.graph_conv2(xx, edge_index, edge_weight)
        # 返回加和后的结果
        return x + xx

# 图神经网络求解器
class GNNSolver(nn.Module):
    def __init__(
        self,
        n_features: int,
        depth: int,
        width: int,
    ):
        super().__init__()
        # 设置节点特征数、图卷积块深度和宽度
        self.n_features = n_features
        self.depth = depth
        self.width = width

        # 实例化节点嵌入层
        self.node_embedder = nn.Linear(n_features, width)
        # 创建图卷积块列表
        self.blocks = nn.ModuleList([GraphBlock(width) for _ in range(depth)])
        # 实例化回归层
        self.regressor = nn.Sequential(
            nn.Linear(width, width),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(width, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch_map: Optional[Tensor] = None,
    ) -> Tensor:
        # 通过节点嵌入层进行特征映射
        x = self.node_embedder(x)
        # 通过多个图卷积块进行前向传播
        for block in self.blocks:
            x = block(x, edge_index, edge_weight, batch_map)
        # 通过回归层获得最终的解
        solution = self.regressor(x)
        # 移除张量的最后一个维度
        return solution.squeeze(dim=-1)
