from typing import Any, Type, Optional, Tuple, List, Dict

import torch
import torch_scatter
import pytorch_lightning as pl
from torch import Tensor

from nsls.metrics import L1Distance, L2Distance, L2Ratio, VectorAngle
from nsls.losses import CosineDistanceLoss

# 定义神经求解器类，继承自 PyTorch Lightning 模块
class NeuralSolver(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        optimizer: Type[torch.optim.Optimizer],
        lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None,
        **scheduler_kwargs: Dict[str, Any],
    ):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scheduler_kwargs = scheduler_kwargs

        # 初始化模型、损失函数和指标
        self.model = None
        self.criterion = CosineDistanceLoss()
        self.elementwise_metric = torch.nn.L1Loss()
        self.systemwise_metrics = torch.nn.ModuleDict(
            {
                "l2_ratio": L2Ratio(),
                "l2_distance": L2Distance(),
                "l1_distance": L1Distance(),
                "angle": VectorAngle(),
            }
        )

    # 设置模型
    def set_model(self, model: torch.nn.Module):
        self.model = model

    # 前向传播方法
    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch_map: Tensor
    ) -> Tensor:
        # 预处理输入数据
        # 获取b
        b = x[:, 0]
        # 对 b 按 batch_map 分组的最大值汇总
        b_max = torch_scatter.scatter(b.abs(), batch_map, reduce="max")
        # 提取边索引对应的 batch_map 中的值
        edge_batch_map = batch_map[edge_index[0]]
        # 对边权重按 edge_batch_map 分组的最大值汇总
        matrix_max = torch_scatter.scatter(
            edge_weight.abs(), edge_batch_map, reduce="max"
        )
        # 对 x 进行归一化，除以相应 batch 中的最大 b 值和边权重最大值
        x[:, 0] /= b_max[batch_map]
        x[:, 1] /= matrix_max[batch_map]
        scaled_weights = edge_weight / matrix_max[edge_batch_map]
        # 模型前向传播
        y_direction = self.model(x, edge_index, scaled_weights, batch_map)
        return y_direction

    # 训练步骤
    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        # 从数据批次中解包数据
        x, edge_index, edge_weight, batch_map, y, b = batch
        # 计算系统的数量
        n_systems = batch_map.max().item() + 1
        # 数据类型转换
        edge_weight = edge_weight.to(torch.float32)
        b = b.to(torch.float32)
        y = y.to(torch.float32)
        # 对输入数据进行前向传播，得到 y 的方向信息
        y_direction = self(x, edge_index, edge_weight, batch_map)
        # 创建稀疏矩阵，表示系统的系数矩阵
        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float32
        )
        # 计算矩阵与 y_direction 的乘积，得到 b 的方向信息
        b_direction = torch.mv(matrix, y_direction)
        # 计算 y_hat 和 y 的loss
        y_loss = self.criterion(y_direction, y, batch_map)
        # 计算 b_hat 和 b 的损失
        b_loss = self.criterion(b_direction, b, batch_map)
        # 总损失为 y_loss 和 b_loss 的和
        loss = y_loss + b_loss
        # 记录训练损失和指标
        self.log("loss/train_solution", y_loss, batch_size=n_systems)
        self.log("loss/train_residual", b_loss, batch_size=n_systems)
        self.log("loss/train", loss, batch_size=n_systems)
        return loss

    # 评估步骤
    def _evaluation_step(
        self,
        phase_name: str,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        # 从数据批次中解包数据
        x, edge_index, edge_weight, batch_map, y, b = batch
        # 计算系统的数量
        n_systems = batch_map.max().item() + 1
        # 创建稀疏矩阵，表示系统的系数矩阵
        matrix = torch.sparse_coo_tensor(
            edge_index, edge_weight, (b.size(0), b.size(0)), dtype=torch.float64
        )
        # 对输入数据进行前向传播，得到 y 的方向信息
        y_direction = self(x, edge_index, edge_weight.to(torch.float32), batch_map)
        y_direction = y_direction.to(torch.float64)
        # 计算矩阵与 y_direction 的乘积，得到 p 的方向信息
        p_direction = torch.mv(matrix, y_direction)
        # 计算 p 的平方范数，使用 scatter_sum 将其按 batch_map 进行汇总
        p_squared_norm = torch_scatter.scatter_sum(p_direction.square(), batch_map)
        # 计算 p 与 b 的点积，同样使用 scatter_sum 按 batch_map 进行汇总
        bp_dot_product = torch_scatter.scatter_sum(p_direction * b, batch_map)
        # 计算缩放因子 scaler
        scaler = torch.clamp_min(bp_dot_product / p_squared_norm, 1e-16)
        # 根据缩放因子对 y_direction 进行缩放，得到 y_hat
        y_hat = y_direction * scaler[batch_map]
        # 根据缩放因子对 p_direction 进行缩放，得到 b_hat
        b_hat = p_direction * scaler[batch_map]
        # 计算 y_hat 和 y 的loss
        y_loss = self.criterion(y_hat, y, batch_map)
        # 计算 b_hat 和 b 的损失
        b_loss = self.criterion(b_hat, b, batch_map)
        # 总损失为 y_loss 和 b_loss 的和
        loss = y_loss + b_loss
        # 记录评估损失和指标
        self.log(f"loss/{phase_name}_solution", y_loss, batch_size=n_systems)
        self.log(f"loss/{phase_name}_residual", b_loss, batch_size=n_systems)
        self.log(f"loss/{phase_name}", loss, batch_size=n_systems)
        # 记录系统级别指标
        for metric_name, metric in self.systemwise_metrics.items():
            self.log(
                f"metrics/{phase_name}_{metric_name}",
                metric(y_hat, y, batch_map),
                batch_size=n_systems,
            )
            self.log(
                f"residual/{phase_name}_{metric_name}",
                metric(b_hat, b, batch_map),
                batch_size=n_systems,
            )
        # 记录元素级别指标
        self.log(
            f"metrics/{phase_name}_absolute_error",
            self.elementwise_metric(y_hat, y),
            batch_size=y.size(0),
        )

    # 验证步骤
    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("val", batch, batch_idx)

    # 测试步骤
    def test_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        self._evaluation_step("test", batch, batch_idx)

    # 配置优化器和学习率调度器
    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        # 初始化优化器和学习率调度器的列表
        optimizers = []
        schedulers = []
        # 使用传入的超参数配置,创建优化器
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # 将优化器添加到列表中
        optimizers.append(optimizer)
        # 如果配置了学习率调度器，则创建学习率调度器，并将其添加到列表中
        if self.lr_scheduler is not None:
            schedulers.append(
                {
                    "scheduler": self.lr_scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "name": "lr",
                }
            )
        return optimizers, schedulers
