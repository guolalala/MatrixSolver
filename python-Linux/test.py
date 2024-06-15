from SparseMatrixSolver import nicslu
from SparseMatrixSolver import eigen
from SparseMatrixSolver import OptimizedSolver
from SparseMatrixSolver import klu
from SparseMatrixSolver import glu

# nicslu.NICSLUSolve("./cz40948.mtx")
# eigen.LLTSolve("./cz1268.mtx")
# OptimizedSolver.OptimizedSolve("./cz1268.mtx")
# klu.KLUSolve("./cz1268.mtx")
# eigen.GaussSolve("./cz1268.mtx")
glu.GLUSolve('./datasets/add20.mtx')