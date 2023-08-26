import os
from ctypes import *

module_path = os.path.dirname(__file__)
dll_path = os.path.join(module_path, "nicsluDll.dll")
nicslu=CDLL(dll_path)

def NICSLUSolve(A,B,X):
    nicslu.NICSLUSolve(A.encode(),B.encode(),X.encode())

if __name__ == '__main__':
    NICSLUSolve("add20.mtx", "add20_b.mtx", "NICSLU.log")