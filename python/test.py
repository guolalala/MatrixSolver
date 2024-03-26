from Matrixsolver import nicslu
from Matrixsolver import eigen
# nicslu.NICSLUSolve("../datasets/ACTIVSg10K.mtx", "../datasets/ACTIVSg10K_b.mtx", "../logs/nicslu.log")

eigen.LDLTSolve("../datasets/ACTIVSg10K.mtx", "../datasets/ACTIVSg10K_b.mtx", "../logs/eigen_LLT.log")

# # 生成ACTIVSg10k_b.mtx文件 n=20000， m=1 元素全为1

# fp = open("../datasets/ACTIVSg10K_b.mtx", "w")
# fp.write("%%MatrixMarket matrix coordinate real general\n%\n")
# fp.write(str(20000) + " " + str(1) + " " + str(20000) + "\n")
# for i in range(20000):
#     fp.write( str(1)  + "\n")
# fp.close()