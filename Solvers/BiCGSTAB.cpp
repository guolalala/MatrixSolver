#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <ctime>
#include <vector>
#include "fstream"
#include "algorithm"

using namespace std;
using namespace Eigen;

int main()
{
    //构造稀疏矩阵


    //数据集
    ifstream fin("../datasets/ACTIVSg10k.mtx");
    if(!fin)
    {
        cout<<"File Read Failed!"<<endl;
        return 1;
    }
    ofstream fout("../logs/BiCGSTAB_10k.log", ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if(!fout)
    {
        cout<<"File Open Failed!"<<endl;
        return 1;
    }

    int M, N, L;
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');
    fin >> M >> N >> L;
    
    SparseMatrix<double> A(M, N);
    A.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    fin.close();

    A.setFromTriplets(tripletlist.begin(), tripletlist.end());
    A.makeCompressed();

    //构造右端项
    VectorXd b(M);
    for(int i = 0; i < M; ++i) {
        b(i) = i + 1;
    }

    clock_t  time_stt;

    time_stt = clock();
    BiCGSTAB<SparseMatrix<double>> solver;

    //需要设置最大迭代次数，大概在10w左右能迭代出来
	solver.setMaxIterations(1200000);
	//solver.setTolerance(1e-2);
    solver.compute(A);

    if( solver.info()!=Success)
    {
        cout<<"Decomposition Failed!"<<endl;
        return 1;
    }
    double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    time_stt = clock();
    
    VectorXd x;
    x= solver.solve(b);
    fout << "#iterations:     " << solver.iterations() << endl;
	fout << "estimated error: " << solver.error()      << endl;
	cout << "#iterations:     " << solver.iterations() << endl;
	cout << "estimated error: " << solver.error()      << endl;
    if( solver.info()!=Success)
    {
        cout<<"Solving Failed!"<<endl;
        return 1;
    }

    double solve_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    fout<<"BiCGSTAB for ACTIVSg10k Succeed!"<<endl;
    fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    

    // 计算残差向量的范数
    VectorXd residual = (A*x)-(b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    fout<< "l1Norm norm: " << l1Norm << endl;
    fout<< "Euclidean norm: " << residualNorm << endl;
    fout<< "infinityNorm norm: " << infinityNorm << endl;
    fout<<"x:"<<x<<"\n"<<endl;

    cout<<"BiCGSTAB for ACTIVSg10k Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "l1Norm norm: " << l1Norm << endl;
    cout<< "Euclidean norm: " << residualNorm << endl;
    cout<< "infinityNorm norm: " << infinityNorm << endl;

    fout.close();
    return 0;
}