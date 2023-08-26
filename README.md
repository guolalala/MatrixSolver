# Sparse Matrix Solver repository
通过pip install 相对路径或绝对路径，添加MatrixSolver-0.0.1-py3-none-any.whl软件包。
import MatrixSolver库，包含eigen和nicslu两部分。
eigen包含7种矩阵算法，分别为LLT,LDLT,LU,QR,CG,LSCG,BICG；
nicslu包含NICSLU算法。
算法对应的函数名为***Solve;
函数需要三个bytes类型参数，分别为存放矩阵A的mtx数据集的地址，存放矩阵B的mtx数据集的地址，保存运算结果矩阵X的文件地址；
函数会输出算法的分析、求解时间，计算误差和求解结果。