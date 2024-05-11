from pathlib import Path
from typing import Any, Union, Tuple, Dict

import yaml
import torch
import torch.nn
import torch_geometric.data
import torch_geometric.loader
from torch import Tensor
from torch.utils.data import DataLoader

from nsls import gnn
from nsls import preprocessors
from nsls import augmentations
from nsls.graph_system_dataset import GraphSystemDataset

# 自定义的数据加载器，用于将PyTorch Geometric的Batch解包
class UnpackingCollater(torch_geometric.loader.dataloader.Collater):
    def __init__(self):
        super().__init__(follow_batch=tuple(), exclude_keys=tuple())

    def __call__(
        self, batch: torch_geometric.data.Data
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # 调用父类的 __call__ 方法，获得经过整理的 batch
        collated_batch = super().__call__(batch)
        # 判断整理后的 batch 是否为 PyTorch Geometric 的 Batch 类型
        if isinstance(collated_batch, torch_geometric.data.Batch):
            # 返回解包后的数据项
            return (
                collated_batch.x,
                collated_batch.edge_index,
                collated_batch.edge_attr,
                collated_batch.batch,
                collated_batch.y,
                collated_batch.b,
            )
        # 如果不是 Batch 类型，则直接返回整理后的 batch
        return collated_batch

# 配置类，用于加载和解析模型配置文件
class Config:
    def __init__(self, config_path: Union[str, Path]):
        # 将配置文件路径扩展为绝对路径，并使用yaml库加载配置信息
        self.config_path = Path(config_path).expanduser()
        with self.config_path.open("r") as f:
            self.config = yaml.safe_load(f)

    # 从配置文件中获取训练或测试数据集
    def _get_dataset(self, train: bool) -> GraphSystemDataset:
        dataset_config = self.config["DATASET"]
        split_dataset_config = dataset_config["TRAIN" if train else "TEST"]
        dataset_augmentations = []
        # 检查是否存在数据增强配置
        if "AUGMENTATIONS" in dataset_config:
            augmentations_config = dataset_config["AUGMENTATIONS"]
            for augmentation_config in augmentations_config:
                # 获取数据集增强的类和参数
                augmentation_class = getattr(augmentations, augmentation_config["NAME"])
                augmentation_kwargs = {
                    k.lower(): v for k, v in augmentation_config.items() if k != "NAME"
                }
                augmentation = augmentation_class(**augmentation_kwargs)
                dataset_augmentations.append(augmentation)
        # 返回创建的 GraphSystemDataset 实例
        return GraphSystemDataset(
            dataset_dir=split_dataset_config["DIRECTORY"],
            num_matrices=split_dataset_config["NUM_MATRICES"],
            feature_augmentations=dataset_augmentations,
        )

    # 获取数据预处理器
    def get_preprocessors(self) -> Tuple[preprocessors.Preprocessor, ...]:
        # 检查是否存在数据增强配置
        if "AUGMENTATIONS" not in self.config["DATASET"]:
            # 若不存在，返回空的元组
            return tuple()
        augmentations_config = self.config["DATASET"]["AUGMENTATIONS"]
        processors = []
        for augmentation_config in augmentations_config:
            # 获取数据预处理器的类和参数
            preprocessor_class = getattr(
                preprocessors,
                augmentation_config["NAME"].replace("Augmentation", "Preprocessor"),
            )
            preprocessor_kwargs = {
                k.lower(): v for k, v in augmentation_config.items() if k != "NAME"
            }
            processor = preprocessor_class(**preprocessor_kwargs)
            processors.append(processor)
        # 返回数据预处理器的元组
        return tuple(processors)

    # 获取训练数据集
    def get_train_dataset(self) -> GraphSystemDataset:
        return self._get_dataset(train=True)

    # 获取测试数据集
    def get_test_dataset(self) -> GraphSystemDataset:
        return self._get_dataset(train=False)

    # 获取数据加载器
    def _get_loader(self, train: bool) -> DataLoader:
        dataset = self._get_dataset(train)
        # 根据训练状态确定 batch_size
        if train:
            batch_size = self.config["OPTIMIZER"]["BATCH_SIZE"]
        else:
            batch_size = self.config["TEST"]["BATCH_SIZE"]
        # 创建 DataLoader 实例
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=UnpackingCollater(),
            # Rule of thumb: num_workers = 4 * n_gpus
            # see https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
            num_workers=4,
            pin_memory=True,
        )
        # 返回 DataLoader 实例
        return loader

    # 获取训练数据加载器
    def get_train_loader(self) -> DataLoader:
        return self._get_loader(train=True)

    # 获取测试数据加载器
    def get_test_loader(self) -> DataLoader:
        return self._get_loader(train=False)

    # 获取模型优化器和学习率调度器的参数
    def get_module_params(self) -> Dict[str, Any]:
        optimizer_config = self.config["OPTIMIZER"]
        # 将优化器配置中的参数组装成字典
        params = {
            k.lower(): v
            for k, v in optimizer_config.items()
            if k not in ("NAME", "BATCH_SIZE", "EPOCHS")
        }
        # 添加优化器类到字典
        params["optimizer"] = getattr(torch.optim, optimizer_config["NAME"])
        # 检查是否存在学习率调度器配置
        if "SCHEDULER" in self.config:
            scheduler_config = self.config["SCHEDULER"]
            # 添加学习率调度器类到字典
            params["lr_scheduler"] = getattr(
                torch.optim.lr_scheduler, scheduler_config["NAME"]
            )
            # 将调度器的参数组装到字典
            scheduler_kwargs = {
                k.lower(): v for k, v in scheduler_config.items() if k != "NAME"
            }
            params.update(scheduler_kwargs)
        # 返回参数字典
        return params

    # 获取训练轮数
    def get_epochs(self) -> int:
        return self.config["OPTIMIZER"]["EPOCHS"]

    # 获取模型
    def get_model(self, n_features: int) -> torch.nn.Module:
        architecture_config = self.config["ARCHITECTURE"]
        # 获取 GNN 模型的类
        model_class = getattr(gnn, architecture_config["NAME"])
        model_kwargs = {
            k.lower(): v for k, v in architecture_config.items() if k != "NAME"
        }
        # 添加输入特征数到参数字典
        model_kwargs["n_features"] = n_features
        # 创建 GNN 模型实例
        model = model_class(**model_kwargs)
        # 返回创建的模型实例
        return model

    # 设置学习率
    def set_learning_rate(self, lr: float) -> None:
        self.config["OPTIMIZER"]["LEARNING_RATE"] = lr

    # 保存配置信息到指定目录
    def save(self, output_dir: Union[str, Path]) -> None:
        # 将输出目录路径扩展为绝对路径，创建目录
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        # 构造保存配置文件的路径
        output_path = output_dir / self.config_path.name
        # 使用 yaml 库保存配置信息到文件
        with output_path.open("w") as f:
            yaml.safe_dump(
                self.config, f, default_flow_style=False, allow_unicode=True, indent=4
            )
