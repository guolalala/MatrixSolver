#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from nsls.config import Config
from nsls.config_trainer import ConfigTrainer
from nsls.neural_solver import NeuralSolver
from nsls.single_inference import SingleInference

# 命令行接口类
class CLI:
    def __init__(self):
        # 初始化命令行参数
        parser = argparse.ArgumentParser(
            description="Command line interface for Neural Sparse Linear Solvers",
            usage=(
                "python3 -m nsls <command> [<args>]\n"
                "\n"
                "train       Train the model\n"
                "eval        Evaluate the model\n"
                "export      Export a trained model\n"
            ),
        )
        parser.add_argument(
            "command",
            type=str,
            help="Sub-command to run",
            choices=(
                "train",
                "eval",
                "export",
            ),
        )

        # 解析命令行参数
        args = parser.parse_args(sys.argv[1:2])
        command = args.command.replace("-", "_")
        if not hasattr(self, command):
            print("Unrecognized command")
            parser.print_help()
            exit(1)
        getattr(self, command)()

    @staticmethod
    def train() -> None:
        warnings.filterwarnings(
            "ignore",
            ".*Trying to infer the `batch_size` from an ambiguous collection.*",
        )
        # 初始化训练子命令的参数解析器
        parser = argparse.ArgumentParser(
            description="Train the model",
            usage="python3 -m nsls train config-path [--output-dir OUTPUT-DIR]",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--output-dir", help="Output directory", default="./runs")
        args = parser.parse_args(sys.argv[2:])

        # 从配置文件加载配置信息
        config = Config(args.config_path)

        # 创建配置训练器
        config_trainer = ConfigTrainer(
            config,
            Path(args.output_dir).expanduser(),
            gpus=1 if torch.cuda.is_available() else 0,
        )
        # 根据训练数据的特征维度获取模型
        model = config.get_model(config_trainer.input_dim)
        # 将配置信息保存到训练日志目录
        config.save(config_trainer.trainer.logger.log_dir)
        # 开始模型训练
        config_trainer.fit(model)

    @staticmethod
    def eval() -> None:
        # 评估子命令的参数解析器
        parser = argparse.ArgumentParser(
            description="Evaluate the model",
            usage="python3 -m nsls eval config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        args = parser.parse_args(sys.argv[2:])

        # 从配置文件加载配置信息
        config = Config(args.config_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 从检查点文件中加载相关信息
        checkpoint = torch.load(args.checkpoint, map_location=device)
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
        print(results)

    @staticmethod
    def export() -> None:
        # 导出子命令的参数解析器
        parser = argparse.ArgumentParser(
            description="Export a trained model",
            usage="python3 -m nsls export config-path --checkpoint CHECKPOINT",
        )
        parser.add_argument(
            "config_path",
            metavar="config-path",
            help="Path to configuration file",
        )
        parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
        parser.add_argument(
            "--output-path", help="Output directory", default="model.pt"
        )
        parser.add_argument("--gpu", help="Export model for GPU", action="store_true")
        args = parser.parse_args(sys.argv[2:])

        # 从配置文件加载配置信息
        config = Config(args.config_path)
        device = torch.device("cuda:0") if args.gpu else torch.device("cpu")
        # 加载模型检查点
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # 获取测试数据集
        test_dataset = config.get_test_dataset()
        # 获取模型
        model = config.get_model(test_dataset.feature_dim)
        # 获取数据预处理器
        processors = config.get_preprocessors()
        # 创建用于单个推理的模块
        module = SingleInference(model, processors)
        # 加载模型参数
        module.load_state_dict(checkpoint["state_dict"])
        # 将模块移动到指定设备，设置为评估模式（不进行梯度计算）
        module = module.to(device).eval().requires_grad_(False)
        # 获取测试数据集的样本1121212
        test_sample = test_dataset[0]
        # 将测试数据集样本转换为模型所需的输入格式，并移动到指定设备
        test_inputs = (
            test_sample.b.to(device),
            test_sample.edge_index.to(device),
            test_sample.edge_attr.to(device),
        )
        # 对模块进行追踪（tracing），将其转换为 TorchScript 形式
        traced_module = torch.jit.trace(
            module,
            test_inputs,
        )
        # 冻结追踪后的模块，使其不可再训练
        traced_module = torch.jit.freeze(traced_module)
        # 保存导出的模型
        traced_module.save(args.output_path)

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
    
    dir_path = os.path.dirname(outfile_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f = open(outfile_path, "w")
    print(results, file=f)
if __name__ == "__main__":
    # nsls("./config/nsls_stand_small_128.yaml","./checkpoints/epoch=49-step=312499.ckpt","./logs/nsls.log")
    CLI()
