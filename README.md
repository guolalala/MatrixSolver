# 说明文档

MatrixSolver库是一个在python上使用的用于进行高阶稀疏矩阵快速求解加速的一个本地库。它包含了原本是c++的eigen库中的稀疏矩阵的解法，包括直接求解和间接求解等；还包含了NICSLU算法，一个高性能求解稀疏矩阵的库，主要应用场景是并联电路仿真。

## 环境配置

通过pip install，添加MatrixSolver-0.0.1-py3-none-any.whl软件包，可以使用相对路径或者绝对路径。

以相对路径为例，

```sh
pip install .\dist\MatrixSolver-0.0.1-py3-none-any.whl
```

如果出现这样的输出结果，说明环境配置成功。

```sh
Installing collected packages: MatrixSolver
Successfully installed MatrixSolver-0.0.1
```

可以在python已安装的软件包中查找MatrixSolver，以确保配置成功。

## 模块

MatrixSolver库包含两个模块，分别是eigen和nicslu，对应两类不同的矩阵解法。
通过如下代码可以引入需要使用的模块

```sh
from Matrixsolver import eigen
from Matrixsolver import nicslu
```

## eigen

eigen模块包含7个函数，即7种不同的矩阵解法

```sh
def LLTSolve(A, B, X)
def LDLTSolve(A, B, X)
def LUSolve(A, B, X)
def QRSolve(A, B, X)
def CGSolve(A, B, X)
def LSCGSolve(A, B, X)
def BICGSolve(A, B, X)
```

函数的三个参数分别为存放矩阵A的mtx类型数据集文件的地址，存放矩阵B的mtx类型数据集文件的地址，保存运算结果的文件地址。

以LLTSolve函数为例，
```sh
from Matrixsolver import eigen

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

第二部分内容会输出到所提供的保存运算结果的文件中，除了直接输出的结果，还包括求出的解的具体数据。

```sh
LLTSolver for add20.mtx and add20_b.mtx Solving Succeed!
Compute time: 292 ms
Solve time: 12 ms

Total time: 304 ms

l1Norm norm: 5.04952e-24
Euclidean norm: 1.60906e-24
infinityNorm norm: 1.265e-24
x: -2.5915e-13
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
from Matrixsolver import nicslu

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

第二部分内容会输出到所提供的保存运算结果的文件中，除了直接输出的结果，还包括求出的解的具体数据。

```sh
NICSLUSolver for add20.mtx and add20_b.mtx Solving Succeed!
analysis time: 0.0035882
best ordering method: AMD
factor time: 0.0005245
solve time: 3.64e-05
residual RMSE: 7.53259e-26

-2.5915e-13
4.52418e-13
-3.90866e-12
8.06483e-13
 ……
 ```
