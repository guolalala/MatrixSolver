import os
from ctypes import *

module_path = os.path.dirname(__file__)
dll_path = os.path.join(module_path, "eigenDll.dll")
eigen=CDLL(dll_path)

def LLTSolve(A,B,X):
    eigen.LLTSolve(A.encode(),B.encode(),X.encode())

def LDLTSolve(A,B,X):
    eigen.LDLTSolve(A.encode(),B.encode(),X.encode())

def LUSolve(A,B,X):
    eigen.LUSolve(A.encode(),B.encode(),X.encode())

def QRSolve(A,B,X):
    eigen.QRSolve(A.encode(),B.encode(),X.encode())

def CGSolve(A,B,X):
    eigen.CGSolve(A.encode(),B.encode(),X.encode())

def LSCGSolve(A,B,X):
    eigen.LSCGSolve(A.encode(),B.encode(),X.encode())

def BICGSolve(A, B, X):
    eigen.BICGSolve(A.encode(), B.encode(), X.encode())

if __name__ == '__main__':
    LLTSolve("add20.mtx", "add20_b.mtx", "LLT.log")