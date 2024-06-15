import argparse
from SparseMatrixSolver import nicslu
from SparseMatrixSolver import eigen
from SparseMatrixSolver import OptimizedSolver
from SparseMatrixSolver import klu
from SparseMatrixSolver import glu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse Matrix Solver')
    parser.add_argument('-s','--solver', type=str, choices=['LLTSolve', 'LDLTSolve', 'LUSolve', 'QRSolve', 'CGSolve', 'LSCGSolve', 'BICGSolve', 'JacobiSolve', 'GaussSeidelSolve', 'NewtonSolve', 'GaussSolve', 'KLUSolve', 'NICSLUSolve', 'OptimizedSolve','GLUSolve'], default='optimizedsolve', help='Solver to use')
    parser.add_argument('-m', '--matrix', type=str,default='./datasets/add20.mtx', help='Matrix in the form of a string')
    parser.add_argument('-b', '--b', type=str,default=None, help='Right-hand side vector in the form of a string')
    parser.add_argument('-o', '--output', type=str,default='./output_x.dat', help='Output file name')

    args = parser.parse_args()

    A = args.matrix
    B = args.b
    X = args.output

    if args.solver.lower() == 'optimizedsolve':
        OptimizedSolver.OptimizedSolve(A,B,X)

    elif args.solver.lower() == 'nicslusolve':
        nicslu.NICSLUSolve(A,B,X)

    elif args.solver.lower() == 'klusolve':
        klu.KLUSolve(A,B,X)

    elif args.solver.lower() == 'lltsolve':
        eigen.LLTSolve(A,B,X)

    elif args.solver.lower() == 'ldltsolve':
        eigen.LDLTSolve(A,B,X)

    elif args.solver.lower() == 'lusolve':
        eigen.LUSolve(A,B,X)

    elif args.solver.lower() == 'qrsolve':
        eigen.QRSolve(A,B,X)

    elif args.solver.lower() == 'cgsolve':
        eigen.CGSolve(A,B,X)

    elif args.solver.lower() == 'lscgsolve':
        eigen.LSCGSolve(A,B,X)

    elif args.solver.lower() == 'bicgsolve':
        eigen.BICGSolve(A,B,X)

    elif args.solver.lower() == 'jacobisolve':
        eigen.JacobiSolve(A,B,X)

    elif args.solver.lower() == 'gaussseidelsolve':
        eigen.GaussSeidelSolve(A,B,X)

    elif args.solver.lower() == 'newtonsolve':
        eigen.NewtonSolve(A,B,X)

    elif args.solver.lower() == 'gausssolve':
        eigen.GaussSolve(A,B,X)
    
    elif args.solver.lower() == 'glusolve':
        glu.GLUSolve(A,B,X)

    else:
        print("Invalid solver")
        print("Available solvers: 'LLTSolve', 'LDLTSolve', 'LUSolve', 'QRSolve', 'CGSolve', 'LSCGSolve', 'BICGSolve', 'JacobiSolve', 'GaussSeidelSolve', 'NewtonSolve', 'GaussSolve', 'KLUSolve', 'NICSLUSolve', 'GLUSolve', 'OptimizedSolve'")