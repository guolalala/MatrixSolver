from SparseMatrixSolver import nicslu
from SparseMatrixSolver import eigen
from SparseMatrixSolver import OptimizedSolver

# nicslu.NICSLUSolve("./datasets/ACTIVSg10K.mtx")
# eigen.LLTSolve("./datasets/ACTIVSg10K.mtx")
OptimizedSolver.Solve("./datasets/ACTIVSg10K.mtx")