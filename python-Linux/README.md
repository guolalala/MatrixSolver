# 说明文档

SparseMatrixSolver库是一个在python上使用的用于进行高阶稀疏矩阵快速求解加速的一个本地库。它包含了原本是c++的eigen库中的稀疏矩阵的解法，包括直接求解和间接求解等；还包含了NICSLU算法、KLU算法和加速求解算法等。

## 环境配置

通过pip install，添加[SparseMatrixSolver-0.0.4-py3-none-any.whl](dist/SparseMatrixSolver-0.0.4-py3-none-any.whl)软件包，可以使用相对路径或者绝对路径。

以相对路径为例，

```sh
python -m pip install ./dist/SparseMatrixSolver-0.0.4-py3-none-any.whl
```

如果出现这样的输出结果，说明环境配置成功。

```sh
Processing ./dist/SparseMatrixSolver-0.0.4-py3-none-any.whl
Installing collected packages: SparseMatrixSolver
Successfully installed SparseMatrixSolver-0.0.2
```

可以在python已安装的软件包中查找SparseMatrixSolver，以确保配置成功。

## 模块

MatrixSolver库包含四个模块，分别是eigen，klu，nicslu和OptimizedSolver，分别对应传统求解算法、KLU高速求解算法、NICSLU高速求解算法和加速求解算法。
通过如下代码可以引入需要使用的模块

```sh
from SparseMatrixSolver import nicslu
from SparseMatrixSolver import eigen
from SparseMatrixSolver import OptimizedSolver
from SparseMatrixSolver import klu
from SparseMatrixSolver import glu
```

## eigen

eigen模块包含11个函数，即11种不同的矩阵解法

```sh

# 矩阵分解
def LLTSolve(A, B, X)
def LDLTSolve(A, B, X)
def LUSolve(A, B, X) # Sparse分解
def QRSolve(A, B, X) #Cholesky分解

# 高斯消元法
def GaussSolve(A, B, X) # 高斯消元法

# 迭代法
def GaussSeidelSolve(A, B, X) # 高斯-赛德尔迭代法
def JacobiSolve(A, B, X) # 雅克比迭代法
def NewtonSolve(A, B, X) # 牛顿法
def CGSolve(A, B, X) # CG迭代法
def LSCGSolve(A, B, X) # LSCG迭代法

def BICGSolve(A, B, X) # 拟牛顿迭代法
```

函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

以LLTSolve函数为例，
```sh
from SparseMatrixSolver import eigen

eigen.LLTSolve("add20.mtx", "add20_b.mtx", "LLT.log")
```
函数包含两部分输出结果，第一部分内容会直接输出到控制台界面中，包括运行时间，计算误差等内容，

```sh
LLTSolver for add20.mtx and add20_b.mtx Solving Succeed!
Compute time: 274 ms
Solve time: 13 ms

Total time: 287 ms

l1Norm norm: 5.04952e-24
Euclidean norm: 1.60906e-24
infinityNorm norm: 1.265e-24
```

第二部分内容会输出到所提供的保存运算结果的文件中，包括求出的解的具体数据。

```sh
-2.5915e-13
 4.52418e-13
-3.90866e-12
 8.06483e-13
-8.15546e-12
 -2.5687e-14
 ……
 ```
## nicslu

nicslu模块包含的函数为NICSLU算法

```sh
def NICSLUSolve(A, B, X)
```

函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

```sh
from SparseMatrixSolver import nicslu

nicslu.NICSLUSolve("add20.mtx", "add20_b.mtx", "NICSLU.log")
```
函数包含两部分输出结果，第一部分内容会直接输出到控制台界面中，包括运行时间，计算误差等内容，

```sh
NICSLUSolver for add20.mtx and add20_b.mtx Solving Succeed!
analysis time: 0.0035882
best ordering method: AMD
factor time: 0.0005245
solve time: 3.64e-05
residual RMSE: 7.53259e-26
```

第二部分内容会输出到所提供的保存运算结果的文件中，包括求出的解的具体数据。

```sh
-2.5915e-13
4.52418e-13
-3.90866e-12
8.06483e-13
 ……
 ```


## klu

klu模块包含的函数为KLU算法

```sh
def KLUSolve(A, B, X)
```


函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

```sh    
from SparseMatrixSolver import klu

klu.KLUSolve("add20.mtx", "add20_b.mtx", "KLU.log")
```
函数包含两部分输出结果，第一部分内容会直接输出到控制台界面中，包括运行时间，计算误差等内容，

第二部分内容会输出到所提供的保存运算结果的文件中，包括求出的解的具体数据。

## glu

glu模块包含的函数为GLU算法

```sh
def GLUSolve(A, B, X)
```


函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

```sh    
from SparseMatrixSolver import glu

glu.GLUSolve("add20.mtx", "add20_b.mtx", "GLU.log")
```
函数包含两部分输出结果，第一部分内容会直接输出到控制台界面中，包括运行时间，计算误差等内容，

第二部分内容会输出到所提供的保存运算结果的文件中，包括求出的解的具体数据。

## OptimizedSolver

OptimizedSolver模块包含的函数为加速求解算法

```sh
def OptimizedSolve(A, B, X)
```

函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

```sh
from SparseMatrixSolver import OptimizedSolver

OptimizedSolver.OptimizedSolve("add20.mtx", "add20_b.mtx", "Optimized.log")
```
函数包含两部分输出结果，第一部分内容会直接输出到控制台界面中，包括运行时间，计算误差等内容，

第二部分内容会输出到所提供的保存运算结果的文件中，包括求出的解的具体数据。

## 脚本

我们还提供了一个脚本 main.py ，可以直接运行，命令如下：

```sh
python main.py -s optimizedsolve -m ./datasets/add20.mtx -b ./datasets/add20_b.mtx -o ./output_x.dat
```

其中-s参数指定使用的求解器，-m参数指定矩阵文件地址，-b参数指定右端向量文件地址，-o参数指定输出文件地址。

脚本会自动调用相应的求解器，并将结果输出到指定的文件中。

