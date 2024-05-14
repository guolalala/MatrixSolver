import argparse
import sys
from pathlib import Path
import warnings

import numpy as np
import torch
import pytorch_lightning as pl

from .Neuralsolver.nsls.config import Config
from .Neuralsolver.nsls.config_trainer import ConfigTrainer
from .Neuralsolver.nsls.neural_solver import NeuralSolver
from .Neuralsolver.nsls.single_inference import SingleInference