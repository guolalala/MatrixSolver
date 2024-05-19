from setuptools import setup
import setuptools
setup(name='SparseMatrixSolver',
      version='0.0.4',
      author='cbd',
      description='sparse matrix solver',
      packages=['SparseMatrixSolver'],  # 系统自动从当前目录开始找包
      license="apache 3.0",
      package_data={'SparseMatrixSolver': ['libklu.so','libSolver.so','libeigen.so',  'libni_solver.so']}
      )
