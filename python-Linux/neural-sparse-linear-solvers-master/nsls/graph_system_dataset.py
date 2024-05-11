from typing import Sequence
from pathlib import Path
from collections import namedtuple

import numpy as np
import scipy.sparse
import torch
import torch_geometric.data
import zipfile
from torch.utils.data import Dataset
import scanpy as sc

from nsls.augmentations import FeatureAugmentation

# System数据结构的命名元组
System = namedtuple("System", ["A_indices", "A_values", "b", "x"])

# 图形系统数据集类，继承自PyTorch的Dataset类
class GraphSystemDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_matrices: int,
        feature_augmentations: Sequence[FeatureAugmentation] = tuple(),
    ):
        # 初始化数据集
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.num_matrices = num_matrices
        self.feature_augmentations = tuple(feature_augmentations)

        # 获取所有数据文件的路径
        self._paths = tuple(sorted(self.dataset_dir.glob("*.npz")))
        if len(self._paths) != self.num_matrices:
            raise ValueError("The dataset size differs from the expected one")

    def __len__(self) -> int:
        # 返回数据集的大小
        return self.num_matrices

    def __getitem__(self, idx: int) -> torch_geometric.data.Data:
        # 获取数据集中特定索引的数据
        system = None
        while system is None:
            filepath = self._paths[idx]
            with np.load(str(filepath)) as npz_file:
                # prevent crash from random failures with unknown cause
                try:
                    system = System(**npz_file)
                except zipfile.BadZipFile as e:
                    print(
                        f"Skip file after {type(e).__name__}:", e.with_traceback(None)
                    )
                    idx += 1
                    continue
        # 转换数据类型为float32
        A_values = system.A_values.astype(np.float32)
        b = system.b.astype(np.float32)
        m = scipy.sparse.coo_matrix(
            (A_values, list(system.A_indices)),
            shape=(system.b.size, system.b.size),
            dtype=np.float32,
        )

        # 创建一个特征列表，初始包含输入向量 b 和对角矩阵 m 的对角元素
        features = [b[:, np.newaxis], m.diagonal()[:, np.newaxis]]
        # 遍历特征增强器列表，对输入矩阵 m 和向量 b 进行特征增强
        for augmentation in self.feature_augmentations:
            # 调用特征增强器的 __call__ 方法，获取特征增强后的结果
            augmentation_features = augmentation(m, b)
            # 将特征增强后的结果添加到特征列表中
            features.append(augmentation_features)
        # 将所有特征按列堆叠，得到最终的特征矩阵
        features = np.column_stack(features)

        # 创建PyTorch Geometric的Data对象
        data = torch_geometric.data.Data(
            torch.from_numpy(features),
            edge_index=torch.from_numpy(system.A_indices.astype(np.int64)),
            edge_attr=torch.from_numpy(system.A_values),
            y=torch.from_numpy(system.x),
            b=torch.from_numpy(system.b),
        )

        return data

    @property
    def feature_dim(self) -> int:
        # 返回数据集中特征的维度
        return 2 + sum(
            augmentation.feature_dim for augmentation in self.feature_augmentations
        )
