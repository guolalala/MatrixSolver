# Neural Sparse Linear Solvers

Official implementation of Neural Sparse Linear Solvers.

The repository contains:
- in the directory `nsls`, the Python code to train and export a NSLS;
- in the directory `solvers`, the C++ code to benchmark solvers.

## Model training (Python)

### Prerequisites

The code requires Python 3.8+ and the packages listed in `requirements.txt`.

We recommend to install PyTorch following the instructions at https://pytorch.org/: for example we used the line
```bash
pip3 install torch==1.10.1+cu113 \
    torchvision==0.11.2+cu113 \
    torchaudio==0.10.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Similarly, follow the instructions at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html to install PyTorch Geometric.
In our setting, we used the line
```bash
pip3 install torch-scatter torch-sparse torch-cluster \
    torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

The other dependencies can be installed with
```bash
pip3 install -r requirements.txt
```

### Usage

The help page for the Python package `nsls` can be displayed with
```bash
python3 -m nsls --help
```
The available commands are `train`, `eval` and `export`.
Their documentation can be accessed with
```bash
python3 -m nsls <command> --help
```

### Example

The best small model ($d=32$) on StAnD Small can be trained with
```bash
python3 -m nsls train configs/nsls_stand_small_32.yaml
```
It could be necessary to modify the configuration file to point to the correct location of the dataset changing the `DATASET` field.
The current configuration assumes that the StAnD Small dataset is located in:
- `datasets/stand_small_train` for the training set;
- `datasets/stand_small_test` for the test set.

At the end of training, the model can be traced and exported for the deployment in C++ with
```bash
python -m nsls export runs/version_0/nsls_stand_small_32.yaml \
    --checkpoint runs/version_0/checkpoints/epoch=49-step=312499.ckpt \
    --output-path nsls_32.pt --gpu
```

## Solvers benchmark (C++)

### Prerequisites

The benchmark builds and runs on Linux.
The software depends on a modern C++ compiler that supports C++17, on CMake, CUDA, Eigen, Torch, ViennaCL, gtest and benchmark.

### Configure and Build

To build the solvers, create a build directory and run `cmake` on the source directory.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../solvers
make
```

### Usage

To benchmark the available solvers run
```bash
build/benchmark/solvers_benchmark --benchmark_counters_tabular=true \
    --dataset=<dataset_directory> --model=<traced_model>
```
Notice that there is no space between the sign `=` and the passed values.

If you see this error while running the benchmarks
```
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
```
you might want to disable the CPU frequency scaling while running the benchmark:
```
sudo cpupower frequency-set --governor performance
build/benchmark/solvers_benchmark --benchmark_counters_tabular=true \
    --dataset=<dataset_directory> --model=<traced_model>
sudo cpupower frequency-set --governor powersave
```

### Example

For example, we can test the model exported in the example above with
```
build/benchmark/solvers_benchmark --benchmark_counters_tabular=true \
    --dataset=datasets/stand_small_test --model=nsls_32.pt
```