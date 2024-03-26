import os
from ctypes import *

module_path = os.path.dirname(__file__)
dll_path = os.path.join(module_path, "libeigen.so")
eigen=CDLL(dll_path)

def LLTSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.LLTSolve(A,B,X)

def LDLTSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.LDLTSolve(A,B,X)

def LUSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.LUSolve(A,B,X)

def QRSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.QRSolve(A,B,X)

def CGSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.CGSolve(A,B,X)

def LSCGSolve(A,B=None,X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.LSCGSolve(A,B,X)

def BICGSolve(A, B=None, X=None):
    if A!=None:
        A = A.encode()
    if B!=None:
        B = B.encode()
    if X!=None:
        X = X.encode()
    eigen.BICGSolve(A,B,X)

if __name__ == '__main__':
    LLTSolve("add20.mtx")