import os
from ctypes import *

# Load the shared library
module_path = os.path.dirname(__file__)
dll_path = os.path.join(module_path, "libklu.so")
dll=CDLL(dll_path)

# Call the main_function
def KLUSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    dll.KLUSolve(A,B,X)

if __name__ == '__main__':
    KLUSolve("ACTIVSg10K.mtx", "ACTIVSg10K_b.mtx", "klu.log")