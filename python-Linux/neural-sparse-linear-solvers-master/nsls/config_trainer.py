from typing import Union
from pathlib import Path

import torch.nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar

from nsls.config import Config
from nsls.neural_solver import NeuralSolver

# 定义配置训练器类
class ConfigTrainer:
    def __init__(
        self, config: Config, output_dir: Union[None, Path, str], **trainer_kwargs
    ):
        # 保存配置信息
        self.config = config
        # 判断是否提供输出目录，如果没有则禁用记录器和检查点功能
        if output_dir is None:
            logger = False
            enable_checkpointing = False
        else:
            # 创建 TensorBoard 记录器，将日志保存到指定的输出目录
            logger = TensorBoardLogger(
                str(output_dir),
                name="",
                default_hp_metric=False,
            )
            enable_checkpointing = True
        # 创建神经求解器模块
        self.module = NeuralSolver(**config.get_module_params())
        # 创建 PyTorch Lightning 训练器
        self.trainer = pl.Trainer(
            logger=logger,
            enable_checkpointing=enable_checkpointing,
            callbacks=[LearningRateMonitor(), TQDMProgressBar(refresh_rate=100)],
            max_epochs=config.get_epochs(),
            log_every_n_steps=100,
            num_sanity_val_steps=0,
            benchmark=True,
            detect_anomaly=True,
            **trainer_kwargs,
        )
        # 获取训练和验证数据加载器
        self.train_loader = config.get_train_loader()
        self.val_loader = config.get_test_loader()
        # 获取输入维度
        self.input_dim = self.train_loader.dataset.feature_dim

    # 定义训练方法
    def fit(self, model: torch.nn.Module) -> None:
        # 设置神经求解器模块的模型
        self.module.set_model(model)
        # 使用 PyTorch Lightning 训练器进行训练
        return self.trainer.fit(self.module, self.train_loader, self.val_loader)
