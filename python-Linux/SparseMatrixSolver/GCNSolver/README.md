# GCN神经网络求解器 (NNS)

## 概述

本仓库提供了一个基于图卷积网络的求解器 (NNS)，用于使用神经网络解决复杂的计算问题。该求解器具有高度可配置性，允许用户指定配置文件、ckpt文件和输出文件。

## 环境要求

该代码需要 Python 3.8 及以上版本，并需要 `requirements.txt` 中列出的包。

我们建议按照 https://pytorch.org/ 上的说明安装 PyTorch，例如我们使用了以下命令行：
```bash
pip3 install torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

同样地，按照 https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html 上的说明安装 PyTorch Geometric。在我们的设置中，我们使用了以下命令行：
```bash
pip3 install torch-scatter torch-sparse torch-cluster \
    torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

其他依赖项可以通过以下命令行安装：
```bash
pip3 install -r requirements.txt
```

## 使用方法

使用提供的 `main.py` 脚本运行神经网络求解器。该脚本接受几个命令行参数来指定配置文件、ckpt文件和输出文件。

### 命令行参数

- `--config`：配置文件的路径（默认：`./config/nsls_stand_small_128.yaml`）
- `--checkpoint`：ckpt文件的路径（默认：`./checkpoints/epoch=49-step=312499.ckpt`）
- `--output`：输出文件的路径（默认：`./logs/result.log`）

### 示例

```sh
python main.py --config ./config/nsls_stand_small_128.yaml --checkpoint ./checkpoints/epoch=49-step=312499.ckpt --output ./logs/result.log
```

## 配置文件

配置文件是一个 YAML 文件，用于指定神经网络求解器的各种参数和设置。您可以修改默认的配置文件，或者根据需要创建自己的配置文件，我们在config文件夹中给出了一个例子。通过修改TEST内容中的文件路径和数据数量，可以实现求解。

```sh
    TEST:
        DIRECTORY: stand_small/stand_small_test
        NUM_MATRICES: 1000
```

DIRECTORY的内容需要输入一个文件夹的地址，文件夹的内容是您需要进行求解的文件。值得注意的是，我们所支持的文件格式为.npz，文件内容同时包括系数矩阵A和常数项b。

NUM_MATRICES的内容需要输入一个正整数，代表文件夹中您需要进行求解的文件的数量。

如果您自行训练了模型，请使用训练结果中相应的.yaml文件。

## ckpt

ckpt文件包含特定时期神经网络的保存状态。这些文件允许您恢复训练或使用预训练模型进行推理。

我们在checkpoints文件夹中给出了一个例子，如果您自行训练了模型，请使用训练结果中相应的.ckpt文件。

## 输出文件

输出文件包含求解器的结果，包括日志和其他相关信息。请确保输出文件的指定路径可写。

## 输出结果

函数的输出结果包含多种数据，

```sh
{'loss/test': 0.006019345865540685,
 'loss/test_residual': 0.002755073431270616,
 'loss/test_solution': 0.0032642724342700695,
 'metrics/test_absolute_error': 5.5716315928823775e-06,
 'metrics/test_angle': 0.06299540877275082,
 'metrics/test_l1_distance': 0.011779654946303777,
 'metrics/test_l2_distance': 0.0005224505154400303,
 'metrics/test_l2_ratio': 0.10453575774337448,
 'residual/test_angle': 0.04288284291832115,
 'residual/test_l1_distance': 374647.42078746005,
 'residual/test_l2_distance': 11789.574954652528,
 'residual/test_l2_ratio': 0.042719421684586575}
```

其中，比较关键的数据为'metrics/test_absolute_error'和'metrics/test_l2_ratio'，分别代表求解结果的绝对误差和相对误差。
