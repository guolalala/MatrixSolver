from SparseMatrixSolver import nicslu
from SparseMatrixSolver import eigen
from SparseMatrixSolver import OptimizedSolver
from SparseMatrixSolver import klu

# nicslu.NICSLUSolve("./datasets/ACTIVSg10K.mtx")
# eigen.LLTSolve("./datasets/ACTIVSg10K.mtx")
OptimizedSolver.Solve("./datasets/add20.mtx","./datasets/add20_b.mtx")
# klu.KLUSolve("./datasets/add20.mtx","./datasets/add20_b.mtx")