from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import sparse

# 特征增强的抽象基类
class FeatureAugmentation(metaclass=ABCMeta):
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        # 返回特征维度，在子类中实现
        raise NotImplementedError

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        # 特征增强的主要方法，在子类中实现
        raise NotImplementedError

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式
        return f"{self.__class__.__name__}({self.degree})"

# Arnoldi特征增强
class ArnoldiAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        super().__init__()
        # 设置 Arnoldi 特征增强的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        # 初始化特征列表
        features = []
        # 归一化初始向量
        v = b / np.linalg.norm(b)
        # 多次进行 Arnoldi 迭代
        for _ in range(self.degree):
            # Arnoldi 迭代步骤
            v = m.dot(v)
            v = v / np.linalg.norm(v, ord=np.inf)
            features.append(v)
        # 将特征堆叠起来形成最终特征向量
        features = np.stack(features, axis=-1)
        return features

# Jacobi特征增强
class JacobiAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 Jacobi 预处理器的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree + 1

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        # 获取对角线元素
        diagonal = m.diagonal()
        # 计算偏置项
        bias = b / diagonal
        features = [bias]
        # 计算对角矩阵D
        diagonal_matrix = sparse.dia_matrix(
            (diagonal[np.newaxis, :], [0]), shape=m.shape
        )
        # 计算对角矩阵的逆D-1
        inverse_diagonal_matrix = sparse.dia_matrix(
            (1.0 / diagonal[np.newaxis, :], [0]), shape=m.shape
        )
        # 计算上下三角矩阵的和M
        n_matrix = m - diagonal_matrix
        # 计算D-1*M
        h_matrix = inverse_diagonal_matrix.dot(n_matrix)
        # 初始化
        v = bias
        # 多次进行 Jacobi 迭代
        for _ in range(self.degree):
            # Jacobi 迭代步骤
            v = h_matrix.dot(v) + bias
            features.append(v)
        # 将特征连接起来形成最终特征向量
        features = np.stack(features, axis=-1)
        # 归一化特征向量
        features = features / np.linalg.norm(
            features, ord=np.inf, axis=0, keepdims=True
        )
        return features

# 共轭梯度特征增强
class ConjugateGradientAugmentation(FeatureAugmentation):
    def __init__(self, degree: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置共轭梯度预处理器的度数
        self.degree = degree

    @property
    def feature_dim(self) -> int:
        # 返回特征维度
        return self.degree

    def __call__(self, m: sparse.coo_matrix, b: np.ndarray) -> np.ndarray:
        # 初始化变量
        v = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_squared_norm = np.square(r).sum()
        features = []
        # 多次进行共轭梯度迭代
        for _ in range(self.degree):
            # 共轭梯度迭代步骤
            Ap = m.dot(p)
            alpha = r_squared_norm / (p * Ap).sum()
            v = v + alpha * p
            r = r - alpha * Ap
            r1_squared_norm = np.square(r).sum()
            beta = r1_squared_norm / r_squared_norm
            p = r + beta * p
            r_squared_norm = r1_squared_norm
            features.append(v)
        # 将特征连接起来形成最终特征向量
        features = np.stack(features, axis=-1)
        # 归一化特征向量
        features = features / np.linalg.norm(
            features, ord=np.inf, axis=0, keepdims=True
        )
        return features

    def __repr__(self) -> str:
        # 返回对象的字符串表示形式
        return f"{self.__class__.__name__}({self.degree})"
