# ˵���ĵ�

MatrixSolver����һ����python��ʹ�õ����ڽ��и߽�ϡ�������������ٵ�һ�����ؿ⡣��������ԭ����c++��eigen���е�ϡ�����Ľⷨ������ֱ�����ͼ�����ȣ���������NICSLU�㷨��һ�����������ϡ�����Ŀ⣬��ҪӦ�ó����ǲ�����·���档

## ��������

ͨ��pip install�����MatrixSolver-0.0.1-py3-none-any.whl�����������ʹ�����·�����߾���·����

�����·��Ϊ����

```sh
pip install .\dist\MatrixSolver-0.0.1-py3-none-any.whl
```

���������������������˵���������óɹ���

```sh
Installing collected packages: MatrixSolver
Successfully installed MatrixSolver-0.0.1
```

������python�Ѱ�װ��������в���MatrixSolver����ȷ�����óɹ���

## ģ��

MatrixSolver���������ģ�飬�ֱ���eigen��nicslu����Ӧ���಻ͬ�ľ���ⷨ��
ͨ�����´������������Ҫʹ�õ�ģ��

```sh
from Matrixsolver import eigen
from Matrixsolver import nicslu
```

## eigen

eigenģ�����7����������7�ֲ�ͬ�ľ���ⷨ

```sh
def LLTSolve(A, B, X)
def LDLTSolve(A, B, X)
def LUSolve(A, B, X)
def QRSolve(A, B, X)
def CGSolve(A, B, X)
def LSCGSolve(A, B, X)
def BICGSolve(A, B, X)
```

���������������ֱ�Ϊ��ž���A��mtx�������ݼ��ļ��ĵ�ַ����ž���B��mtx�������ݼ��ļ��ĵ�ַ���������������ļ���ַ��

��LLTSolve����Ϊ����
```sh
from Matrixsolver import eigen

eigen.LLTSolve("add20.mtx", "add20_b.mtx", "LLT.log")
```
������������������������һ�������ݻ�ֱ�����������̨�����У���������ʱ�䣬�����������ݣ�

```sh
LLTSolver for add20.mtx and add20_b.mtx Solving Succeed!
Compute time: 274 ms
Solve time: 13 ms

Total time: 287 ms

l1Norm norm: 5.04952e-24
Euclidean norm: 1.60906e-24
infinityNorm norm: 1.265e-24
```

�ڶ��������ݻ���������ṩ�ı������������ļ��У�����ֱ������Ľ��������������Ľ�ľ������ݡ�

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
 ����
 ```
## nicslu

nicsluģ������ĺ���ΪNICSLU�㷨

```sh
def NICSLUSolve(A, B, X)
```

���������������ֱ�Ϊ��ž���A��mtx�������ݼ��ļ��ĵ�ַ����ž���B��mtx�������ݼ��ļ��ĵ�ַ���������������ļ���ַ��

```sh
from Matrixsolver import nicslu

nicslu.NICSLUSolve("add20.mtx", "add20_b.mtx", "NICSLU.log")
```
������������������������һ�������ݻ�ֱ�����������̨�����У���������ʱ�䣬�����������ݣ�

```sh
NICSLUSolver for add20.mtx and add20_b.mtx Solving Succeed!
analysis time: 0.0035882
best ordering method: AMD
factor time: 0.0005245
solve time: 3.64e-05
residual RMSE: 7.53259e-26
```

�ڶ��������ݻ���������ṩ�ı������������ļ��У�����ֱ������Ľ��������������Ľ�ľ������ݡ�

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
 ����
 ```
