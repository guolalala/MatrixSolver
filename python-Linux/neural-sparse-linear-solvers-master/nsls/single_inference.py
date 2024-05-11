from typing import Tuple

import torch
from torch import Tensor

# 定义单一推断模块
class SingleInference(torch.nn.Module):
     # 初始化函数，接受一个模型对象和一组预处理器对象。
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessors: Tuple[torch.nn.Module, ...] = tuple(),
    ):
        super().__init__()
        # 模型和预处理器
        self.model = model
        self.preprocessors = preprocessors

    def forward(self, b: Tensor, m_indices: Tensor, m_values: Tensor) -> Tensor:
        # Create the sparse matrix
        ##m = torch.sparse.DoubleTensor(m_indices, m_values)
        m = torch.sparse_coo_tensor(m_indices, m_values)

        # Get the diagonal of the sparse matrix
        diagonal_mask = m_indices[0] == m_indices[1]
        diagonal_values = m_values[diagonal_mask]
        diagonal = torch.zeros_like(b)
        diagonal[m_indices[0][diagonal_mask]] = diagonal_values

        # Preprocess the input features
        x = torch.stack([b, diagonal], dim=-1)
        features = [x]
        for preprocessor in self.preprocessors:
            features.append(preprocessor(m, b, diagonal))
        x = torch.cat(features, dim=-1)

        # Rescale input system
        b_max = torch.linalg.vector_norm(b, ord=torch.inf)
        m_max = torch.linalg.vector_norm(m_values, ord=torch.inf)
        x[:, 0] /= b_max
        x[:, 1] /= m_max
        scaled_m_values = m_values / m_max
        # Run the model
        y_direction = self.model(
            x.to(torch.float32), m_indices, scaled_m_values.to(torch.float32)
        )
        # 更改数据类型
        y_direction = y_direction.to(torch.float64)
        # 再次缩放处理
        p_direction = torch.mv(m, y_direction)
        p_squared_norm = p_direction.square().sum()
        bp_dot_product = p_direction.dot(b)
        scaler = torch.clamp_min(bp_dot_product / p_squared_norm, 1e-16)
        y_hat = y_direction * scaler
        return y_hat
