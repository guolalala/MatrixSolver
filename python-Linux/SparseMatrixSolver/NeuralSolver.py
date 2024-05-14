import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl

# 最终是把下面这个Solve函数合进SparseMatrixSolver库里，能够实现
# from SparseMatrixSolver import NeuralSolver
# NeuralSolver.Solve(config_path,checkpoint_path,outfile_path)调用
# stand_small_test里面只保留了一个npz，在README里加上完整数据集的下载链接吧

def Solve(config_path,checkpoint_path,outfile_path):
    # 从配置文件加载配置信息
    config = Config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 从检查点文件中加载相关信息
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 获取测试数据加载器
    test_loader = config.get_test_loader()
    # 根据测试数据集的特征维度获取模型
    model = config.get_model(test_loader.dataset.feature_dim)
    # 初始化NeuralSolver求解器模块，传入超参数
    module = NeuralSolver(**checkpoint["hyper_parameters"])
    # 设置求解器模块的模型
    module.set_model(model)
    # 从检查点中加载预训练模型的状态字典
    module.load_state_dict(checkpoint["state_dict"])
    # 创建PyTorch Lightning的Trainer并进行训练
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
    )
    # 运行测试集并获取测试结果
    results = trainer.test(module, dataloaders=test_loader)
    f = open(outfile_path, "w")
    print(results, file=f)