#include <iostream>
#include <Eigen/Eigen>
#include <cmath>
#include <ctime>
#include <vector>
#include "fstream"
#include "algorithm"

using namespace std;
using namespace Eigen;

#define num 4000

int main()
{
    //构��稀疏矩阄1�7

    //箢�单矩阄1�7
    /*SparseMatrix<double> A(num,num);
    vector<Triplet<double>> tripletlist;
    for(int i = 0; i < num; ++i) {
        tripletlist.push_back(Triplet<double>(i, i, i*i+1));
        tripletlist.push_back(Triplet<double>(i, num-1-i, i*(num-1-i)+1));
    }*/

    //数据雄1�7
    ifstream fin("../datasets/ACTIVSg10k.mtx");
    if(!fin)
    {
        cout<<"File Read Failed!"<<endl;
        return 1;
    }
    ofstream fout("../logs/LLT_10k(T).log", ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数捄1�7
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

    //构��右端项
    VectorXd b(M);
    for(int i = 0; i < M; ++i) {
        b(i) = i + 1;
    }
    //因为llt分解要求A是对称正定的，一般的矩阵不满足这个条件，故构造新的线性方程：(A的转罄1�7*A)*x = （A的转罄1�7*b），此方程与原方程同规1�7
    b = A.transpose()*b;
    A = A.transpose()*A;
    
    clock_t  time_stt;

    //LLT分解
    time_stt = clock();
    SimplicialLLT<SparseMatrix<double>> llt;

    llt.compute( A);

    if(llt.info()!=Success)
    {
        cout<<"Decomposition Failed!"<<endl;
        return 1;
    }
    double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    time_stt = clock();
    
    VectorXd x;
    x=llt.solve( b);
    if(llt.info()!=Success)
    {
        cout<<"Solving Failed!"<<endl;
        return 1;
    }

    double solve_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    fout<<"LLTSolver for ACTIVSg10k(T) Succeed!"<<endl;
    fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    
    //fout<<"Solving time is:"<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC<<"ms"<<endl;
    // 计算残差向量的范敄1�7
    VectorXd residual = ( A*x)-( b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    fout<< "l1Norm norm: " << l1Norm << endl;
    fout<< "Euclidean norm: " << residualNorm << endl;
    fout<< "infinityNorm norm: " << infinityNorm << endl;
    fout<<"x:"<<x<<"\n"<<endl;

    cout<<"LLTSolver for ACTIVSg10k(T) Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "l1Norm norm: " << l1Norm << endl;
    cout<< "Euclidean norm: " << residualNorm << endl;
    cout<< "infinityNorm norm: " << infinityNorm << endl;

    fout.close();
    return 0;
}