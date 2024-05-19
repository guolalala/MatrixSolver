import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl
import sys
import os

# 最终是把下面这个Solve函数合进SparseMatrixSolver库里，能够实现
# from SparseMatrixSolver import NeuralSolver
# NeuralSolver.Solve(config_path,checkpoint_path,outfile_path)调用
# stand_small_test里面只保留了一个npz，在README里加上完整数据集的下载链接吧
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'GCNSolver')))
from GCNSolver.nsls.__main__ import Solve

Solve('./GCNSolver/config/nsls_stand_small_128.yaml', './GCNSolver/checkpoints/epoch=49-step=312499.ckpt', './result.log')