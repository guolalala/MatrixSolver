from setuptools import setup
import setuptools
setup(name='MatrixSolver',
      version='0.0.1',
      author='ltz',
      description='eigen and nicslu',
      packages=['Matrixsolver'],  # 系统自动从当前目录开始找包
      license="apache 3.0",
      package_data={'Matrixsolver': ['nicslu.dll', 'nicsluDll.dll', 'nicslu.lic', 'eigenDLL.dll']}
      )
