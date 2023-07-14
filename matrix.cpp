#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <ctime>
#include <vector>
#include "fstream"
#include "algorithm"

using namespace std;
using namespace Eigen;

#define num 20000

int main()
{
    //构造稀疏矩阵

    //简单矩阵
    /*SparseMatrix<double> A(num,num);
    vector<Triplet<double>> tripletlist;
    for(int i = 0; i < num; ++i) {
        tripletlist.push_back(Triplet<double>(i, i, i*i+1));
        tripletlist.push_back(Triplet<double>(i, num-1-i, i*(num-1-i)+1));
    }*/

    //数据集
    ifstream fin("D:/ACTIVSg10K/ACTIVSg2000.mtx");
    //ifstream fin("D:/ACTIVSg10K/ACTIVSg10K.mtx");
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
    VectorXd b(num);
    for(int i = 0; i < num; ++i) {
        b(i) = i + 1;
    }

    ofstream fout("D:/result3.txt");
    clock_t  time_stt;
/*
    //方式1：LLT分解
    time_stt = clock();
    SimplicialLLT<SparseMatrix<double>> llt;
    //因为llt分解要求A是对称正定的，一般的矩阵不满足这个条件，故构造新的线性方程：(A的转置*A)*x = （A的转置*b），此方程与原方程同解
    llt.compute(A.transpose()*A);
    VectorXd x1;
    x1=llt.solve(A.transpose()*b);
    fout<<"x1 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x1:"<<x1<<"\n"<<endl;
    cout << "x1 finish" << endl;

    //方式2：LDLT分解
    time_stt = clock();
    SimplicialLDLT<SparseMatrix<double>> ldlt; 
    ldlt.compute(A.transpose()*A);
    VectorXd x2;
    x2=ldlt.solve(A.transpose()*b);
    fout<<"x2 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x2:"<<x2<<"\n"<<endl;
    cout << "x2 finish" << endl;

    //方式3：LU分解
    time_stt = clock();
    SparseLU<SparseMatrix<double>> lu;
	lu.compute(A);
    VectorXd x3;
    x3=lu.solve(b);
    fout<<"x3 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x3:"<<x3<<"\n"<<endl;
    cout << "x3 finish" << endl;
*/
    //方式4：QR分解
    time_stt = clock();
    SparseQR<SparseMatrix<double>,AMDOrdering<int>> qr;
	qr.compute(A);
    VectorXd x4;
    x4=qr.solve(b);
    
    cout<<"x4:"<<x4<<"\n"<<endl;
    cout<<"x4 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    cout << "x4 finish" << endl;
/*
    //方式5：共轭梯度迭代求解
    time_stt = clock();
    ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
    cg.setTolerance(1e-8);
    cg.compute(A.transpose()*A);
    VectorXd x5;
    x5 = cg.solve(A.transpose()*b);
    fout<<"x5 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x5:\n"<<x5<<"\n"<<endl;
    cout << "x5 finish" << endl;

    //方式6：最小二乘共轭梯度迭代求解
    time_stt = clock();
    LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
    lscg.setTolerance(1e-8);
    lscg.compute(A);
    VectorXd x6;
    x6 = lscg.solve(b);
    fout<<"x6 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x6:\n"<<x6<<"\n"<<endl;
    cout << "x6 finish" << endl;

    //方式7：稳定双共轭梯度迭代求解
    //time_stt = clock();
    BiCGSTAB<SparseMatrix<double>> solver;
    solver.setTolerance(1e-8);
    solver.compute(A);
    VectorXd x7;
    x7 = solver.solve(b);
    fout<<"x7 time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    fout<<"x7:\n"<<x7<<"\n"<<endl;
*/
    fout.close();
    cout << "x7 finish" << endl;
    return 0;
}