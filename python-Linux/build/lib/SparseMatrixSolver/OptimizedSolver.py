import os
from ctypes import *

# Load the shared library
module_path = os.path.dirname(__file__)
dll_path = os.path.join(module_path, "libSolver.so")
optisolver=CDLL(dll_path)

# Call the main_function
def OptimizedSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    optisolver.solve(A,B,X)

if __name__ == '__main__':
    OptimizedSolve("../datasets/ACTIVSg10K.mtx", "../datasets/ACTIVSg10K_b.mtx", "../NICSLU.log")