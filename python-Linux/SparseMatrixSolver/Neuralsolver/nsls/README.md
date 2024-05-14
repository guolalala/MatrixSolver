# 说明文档

如果您不想自行训练模型，我们为您提供了一个训练完成的模型的例子。

## 环境配置

与neural-sparse-linear-solvers-master\README.md中的环境配置操作相同，如果您已经配置好了，可以直接使用。

## 函数

我们在main.py中定义了nsls函数，可以直接进行求解。

```sh
def nsls(config_path,checkpoint_path,outfile_path)
```

### config_path

其中config_path是一个.yaml文件，输入模型的相关参数，我们在config文件夹中给出了一个例子。通过修改TEST内容中的文件路径和数据数量，可以实现求解。

```sh
    TEST:
        DIRECTORY: stand_small/stand_small_test
        NUM_MATRICES: 1000
```

DIRECTORY的内容需要输入一个文件夹的地址，文件夹的内容是您需要进行求解的文件。值得注意的是，我们所支持的文件格式为.npz，文件内容同时包括系数矩阵A和常数项b。
NUM_MATRICES的内容需要输入一个正整数，代表文件夹中您需要进行求解的文件的数量。

如果您自行训练了模型，请使用训练结果中相应的.yaml文件。

### checkpoint_path

其中config_path是一个.ckpt文件，输入模型的相关参数，我们在checkpoints文件夹中给出了一个例子。

如果您自行训练了模型，请使用训练结果中相应的.ckpt文件。

### outfile_path

其中outfile_path是保存运算结果的文件地址。

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
