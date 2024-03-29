#include <stdio.h>
#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <ctime>
#include <vector>
#include "fstream"
#include "algorithm"

extern "C" __declspec(dllexport) void LDLTSolve(char* A, char* B, char* X);

using namespace std;
using namespace Eigen;


void LDLTSolve(char* A, char* B,char* X)
{
    //read matrix A
    ifstream fin(A);
    if(!fin)
    {
        cout<<"File A Read Failed!"<<endl;
        return;
    }

    int M, N, L;
    while (fin.peek() == '%')
        fin.ignore(2048, '\n');
    fin >> M >> N >> L;
    
    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    fin.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    ifstream bin(B);
    if (!bin)
    {
        cout << "File B Read Failed!" << endl;
        return;
    }
    while (bin.peek() == '%')
        bin.ignore(2048, '\n');
    bin >> M >> N;
    VectorXd b(M);
    for(int i = 0; i < M; ++i) {
        bin >> b(i);
    }
    bin.close();

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File X Open Failed!" << endl;
        return;
    }

    b = a.transpose()*b;
    a = a.transpose()*a;
    
    clock_t  time_stt;

    //LDLT分解
    time_stt = clock();
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(a);

    if(solver.info()!=Success)
    {
        cout<<"Decomposition Failed!"<<endl;
        return;
    }
    double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    time_stt = clock();
    
    VectorXd x;
    x=solver.solve(b);
    if(solver.info()!=Success)
    {
        cout<<"Solving Failed!"<<endl;
        return;
    }
    
    double solve_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    fout<<"LDLTSolver for "<< A <<" and "<< B << " Solving Succeed!" << endl;
    fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;

    // 计算残差向量的范数
    VectorXd residual = a*x-b;
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    fout<< "l1Norm norm: " << l1Norm << endl;
    fout<< "Euclidean norm: " << residualNorm << endl;
    fout<< "infinityNorm norm: " << infinityNorm << endl;

    fout<<"x:"<<x<<"\n"<<endl;

    cout<<"LDLTSolver for "<< A <<" and "<< B <<" Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "l1Norm norm: " << l1Norm << endl;
    cout<< "Euclidean norm: " << residualNorm << endl;
    cout<< "infinityNorm norm: " << infinityNorm << endl;

    fout.close();
    return;
}