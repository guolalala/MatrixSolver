import torch

from torch import Tensor
from torchmetrics import Metric
from torch_scatter import scatter_sum

# 定义 L1 距离指标类
class L1Distance(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 添加状态变量
        self.add_state(
            "distance",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    # 更新状态变量
    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        # 累加预测值和目标之间的 L1 距离
        self.distance += torch.abs(preds - target).sum()
        # 累加当前批次的系统数量
        self.total += batch_map.max() + 1

    # 计算最终指标值
    def compute(self) -> Tensor:
        return self.distance / self.total

# 定义 L2 距离指标类
class L2Distance(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 添加状态变量
        self.add_state(
            "distance",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    # 更新状态变量
    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        # 计算预测值和目标之间的平方差
        squared_difference = torch.square(preds - target)
        # 计算每个系统的平方差和，然后取平方根
        distance = torch.sqrt(scatter_sum(squared_difference, batch_map))
        # 累加平方根距离到状态变量
        self.distance += distance.sum()
        # 累加当前批次的系统数量到状态变量
        self.total += distance.size(0)

    # 计算最终指标值
    def compute(self) -> Tensor:
        return self.distance / self.total

# 定义 L1 比例指标类
class L1Ratio(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # 添加状态变量
        self.add_state(
            "ratio", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    # 更新状态变量
    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        # 计算目标的 L1 范数
        target_norm = scatter_sum(torch.abs(target), batch_map)
        # 计算预测值和目标之间的 L1 距离
        distance = scatter_sum(torch.abs(preds - target), batch_map)
        # 计算 L1 比例，即距离与目标范数的比值
        ratio = distance / target_norm
        # 累加 L1 比例
        self.ratio += ratio.sum()
        # 累加当前批次的系统数量
        self.total += ratio.size(0)

    # 计算最终指标值
    def compute(self) -> Tensor:
        return self.ratio / self.total

# 定义 L2 比例指标类
class L2Ratio(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # 添加状态变量
        self.add_state(
            "ratio", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    # 更新状态变量
    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        # 计算目标的平方和，即每个系统目标值的平方和
        target_norm = scatter_sum(torch.square(target), batch_map)
        # 计算预测值和目标之间的平方差和，然后取平方根
        distance = scatter_sum(torch.square(preds - target), batch_map)
        # 计算 L2 比例，即平方根距离与目标平方和的比值
        ratio = torch.sqrt(distance / target_norm)
        # 累加 L2 比例
        self.ratio += ratio.sum()
        # 累加当前批次的系统数量
        self.total += ratio.size(0)

    # 计算最终指标值
    def compute(self) -> Tensor:
        return self.ratio / self.total

# 定义向量夹角指标类
class VectorAngle(Metric):
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # 添加状态变量
        self.add_state(
            "angle",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float64), dist_reduce_fx="sum"
        )

    # 更新状态变量
    def update(self, preds: Tensor, target: Tensor, batch_map: Tensor) -> None:
        # 定义一个很小的常数
        eps = 1e-12
        # 计算预测值的 L2 范数，即每个系统预测值的平方和的平方根
        preds_norm = scatter_sum(preds.square(), batch_map).sqrt_()
        # 计算目标值的 L2 范数，即每个系统目标值的平方和的平方根
        target_norm = scatter_sum(target.square(), batch_map).sqrt_()
        # 计算预测值和目标值的点积
        dot_product = scatter_sum(preds * target, batch_map)
        # 计算余弦相似度，并通过 clamp_min 方法避免除零错误
        cosine = dot_product / torch.clamp_min(preds_norm * target_norm, eps)
        # 计算向量夹角，使用反余弦函数计算，并将结果累加
        self.angle += torch.arccos(cosine).sum()
        # 累加当前批次的系统数量
        self.total += cosine.size(0)

    # 计算最终指标值
    def compute(self) -> Tensor:
        return self.angle / self.total
