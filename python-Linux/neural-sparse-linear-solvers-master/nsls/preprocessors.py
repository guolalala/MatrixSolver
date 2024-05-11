from abc import ABCMeta, abstractmethod

from torch import Tensor
import torch
import torch.nn.functional as F

# 数据预处理器的抽象基类
class Preprocessor(torch.nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        # 返回特征维度，在子类中实现
        raise NotImplementedError

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        # 前向传播方法，在子类中实现
        raise NotImplementedError

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式
        return f"{self.__class__.__name__}({self.degree})"

# Arnoldi预处理器
class ArnoldiPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 Arnoldi 预处理器的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        # 初始化特征列表
        features = []
        # 归一化初始向量
        v = F.normalize(b, dim=0)
        # 多次进行 Arnoldi 迭代
        for _ in range(self.degree):
            # Arnoldi 迭代步骤
            v = F.normalize(m.mv(v), p=torch.inf, dim=0)
            features.append(v)
        # 将特征堆叠起来形成最终特征向量
        features = torch.stack(features, dim=-1)
        return features

# Jacobi预处理器
class JacobiPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 Jacobi 预处理器的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree + 1

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        # 计算偏置项
        bias = torch.unsqueeze(b / d, dim=1)
        features = [bias]
        # 获取稀疏矩阵的索引和数值
        indices = m._indices()
        h_matrix = torch.sparse_coo_tensor(indices, m._values() / d[indices[0]])
        diagonal_mask = indices[0] == indices[1]
        # 将对角线元素设为0
        h_matrix._values().masked_fill_(diagonal_mask, 0.0)

        v = bias
        # 多次进行 Jacobi 迭代
        for _ in range(self.degree):
            # Jacobi 迭代步骤
            v = torch.sparse.addmm(bias, h_matrix, v)
            features.append(v)
        # 将特征连接起来形成最终特征向量
        features = torch.cat(features, dim=-1)
        # 归一化特征向量
        features = F.normalize(features, p=torch.inf, dim=0)
        return features

# 共轭梯度预处理器
class ConjugateGradientPreprocessor(Preprocessor):
    def __init__(self, degree: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置共轭梯度预处理器的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree

    def forward(self, m: Tensor, b: Tensor, d: Tensor) -> Tensor:
        # 初始化变量
        v = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        r_squared_norm = r.square().sum()
        features = []
        # 多次进行共轭梯度迭代
        for _ in range(self.degree):
            # 共轭梯度迭代步骤
            Ap = m.mv(p)
            alpha = r_squared_norm / (p * Ap).sum()
            v = v + alpha * p
            r = r - alpha * Ap
            r1_squared_norm = r.square().sum()
            beta = r1_squared_norm / r_squared_norm
            p = r + beta * p
            r_squared_norm = r1_squared_norm
            features.append(v)
        # 将特征堆叠起来形成最终特征向量
        features = torch.stack(features, dim=-1)
        # 归一化特征向量
        features = F.normalize(features, p=torch.inf, dim=0)
        return features
