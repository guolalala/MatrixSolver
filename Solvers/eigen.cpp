#include <iostream>
#include <Eigen>
#include <cmath>
#include <ctime>
#include <vector>
#include "fstream"
#include "algorithm"
#include <iomanip>

// extern "C" __declspec(dllexport) void BICGSolve(char* A, char* B, char* X);

using namespace std;
using namespace Eigen;
extern "C"{
void help_message()
{
    cout<<"Traditional Solver"<<endl;
}
void BICGSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if(!ain)
    {
        cout<<"File A Read Failed!"<<endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;

    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }

    clock_t  time_stt;

    time_stt = clock();
    BiCGSTAB<SparseMatrix<double>> solver;

    //需要设置最大迭代次数
    solver.setMaxIterations(1200000);
    //solver.setTolerance(1e-2);
    solver.compute(a);

    if( solver.info()!=Success)
    {
        cout<<"Decomposition Failed!"<<endl;
        return;
    }
    double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    time_stt = clock();

    VectorXd x;
    x= solver.solve(b);
    // fout << "#iterations:     " << solver.iterations() << endl;
    // fout << "estimated error: " << solver.error()      << endl;
    cout << "#iterations:     " << solver.iterations() << endl;
    // cout << "estimated error: " << solver.error()      << endl;
    if( solver.info()!=Success)
    {
        cout<<"Solving Failed!"<<endl;
        return;
    }

    double solve_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    // fout<<"BiCGSTAB for " << A << " Solving Succeed!"<<endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;


    // 计算残差向量的范数
    VectorXd residual = (a*x)-(b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;
    fout<< std::setprecision(15) <<x<<endl;

    cout<<"BiCGSTAB for " << A <<  " Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void CGSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if(!ain)
    {
        cout<<"File A Read Failed!"<<endl;
        return ;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;
    
    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }

    b = a.transpose()*b;
    a = a.transpose()*a;

    clock_t  time_stt;

    time_stt = clock();
    ConjugateGradient<SparseMatrix<double>, Lower|Upper> solver;

    //需要设置最大迭代次数
	solver.setMaxIterations(12000000);
	//solver.setTolerance(1e-2);
    solver.compute(a);

    if( solver.info()!=Success)
    {
        cout<<"Decomposition Failed!"<<endl;
        return;
    }
    double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    time_stt = clock();
    
    VectorXd x;
    x= solver.solve(b);
	// fout << "#iterations:     " << solver.iterations() << endl;
	// fout << "estimated error: " << solver.error()      << endl;
	cout << "#iterations:     " << solver.iterations() << endl;
	cout << "estimated error: " << solver.error()      << endl;
    if( solver.info()!=Success)
    {
        cout<<"Solving Failed!"<<endl;
        return;
    }

    double solve_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
    // fout<<"CgSolver for " << A << " and " << B << " Solving Succeed!"<<endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    

    // 计算残差向量的范数
    VectorXd residual = ( a*x)-( b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;
    fout<< std::setprecision(15) <<x<<endl;

    cout<<"CgSolver for " << A << " Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void LDLTSolve(char* A, char* B,char* X)
{
    //read matrix A
    ifstream fin(A);
    if(!fin)
    {
        cout<<"File A Read Failed!"<<endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
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
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
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
    // fout<<"LDLTSolver for "<< A <<" and "<< B << " Solving Succeed!" << endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;

    // 计算残差向量的范数
    VectorXd residual = a*x-b;
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;

    fout<< std::setprecision(15) <<x<<endl;

    cout<<"LDLTSolver for "<< A <<" Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void LLTSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if(!ain)
    {
        cout<<"File A Read Failed!"<<endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;
    
    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }

    b = a.transpose()*b;
    a = a.transpose()*a;
    
    clock_t  time_stt;

    //LLT鍒嗚В
    time_stt = clock();
    SimplicialLLT<SparseMatrix<double>> solver;

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
    // fout<<"LLTSolver for " << A << " and " << B << " Solving Succeed!"<<endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    
    // 计算残差向量的范数
    VectorXd residual = (a*x)-( b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;
    fout<< std::setprecision(15) <<x<<endl;

    cout<<"LLTSolver for " << A << " Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void LSCGSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if (!ain)
    {
        cout << "File A Read Failed!" << endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;

    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }

    clock_t  time_stt;

    time_stt = clock();
    LeastSquaresConjugateGradient<SparseMatrix<double>> solver;


    solver.setMaxIterations(1200000);
    solver.compute(a);

    if (solver.info() != Success)
    {
        cout << "Decomposition Failed!" << endl;
        return;
    }
    double compute_time = 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC;
    time_stt = clock();

    VectorXd x;
    x = solver.solve(b);
    // fout << "#iterations:     " << solver.iterations() << endl;
    // fout << "estimated error: " << solver.error() << endl;
    cout << "#iterations:     " << solver.iterations() << endl;
    cout << "estimated error: " << solver.error() << endl;
    if (solver.info() != Success)
    {
        cout << "Solving Failed!" << endl;
        return;
    }

    double solve_time = 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC;
    // fout << "LSCGSolver for " << A << " and " << B << " Solving Succeed!" << endl;
    // fout << "Compute time: " << compute_time << " ms" << endl;
    // fout << "Solve time: " << solve_time << " ms" << endl << endl;
    // fout << "Total time: " << compute_time + solve_time << " ms" << endl << endl;


    VectorXd residual = (a * x) - (b);
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout << "l1Norm norm: " << l1Norm << endl;
    // fout << "Euclidean norm: " << residualNorm << endl;
    // fout << "infinityNorm norm: " << infinityNorm << endl;
    fout<< std::setprecision(15) <<x<<endl;

    cout << "LSCGSolver for " << A << " Solving Succeed!" << endl;
    cout << "Compute time: " << compute_time << " ms" << endl;
    cout << "Solve time: " << solve_time << " ms" << endl << endl;
    cout << "Total time: " << compute_time + solve_time << " ms" << endl << endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void LUSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if (!ain)
    {
        cout << "File A Read Failed!" << endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;

    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }
    
    clock_t  time_stt;

    //LU分解
    time_stt = clock();
    SparseLU<SparseMatrix<double>> solver;
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
    // fout<<"LUSolver for " << A << " and " << B << " Solving Succeed!"<<endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;

    // 计算残差向量的范数
    VectorXd residual = a*x-b;
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;

    fout<< std::setprecision(15) <<x<<endl;

    cout<<"LUSolver for " << A << " Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    
void QRSolve(char* A, char* B, char* X)
{
    //read matrix A
    ifstream ain(A);
    if (!ain)
    {
        cout << "File A Read Failed!" << endl;
        return;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    int M, N, L;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> M >> N >> L;

    SparseMatrix<double> a(M, N);
    a.reserve(L);
    vector<Triplet<double>> tripletlist;
    for (int i = 0; i < L; ++i) {
        int m, n;
        double data;
        ain >> m >> n >> data;
        tripletlist.push_back(Triplet<double>(m - 1, n - 1, data));// m - 1 and n - 1 to set index start from 0
    }
    ain.close();

    a.setFromTriplets(tripletlist.begin(), tripletlist.end());
    a.makeCompressed();

    //read matrix B
    VectorXd b(M);
    if(B!=NULL)
    {
        ifstream bin(B);
        if (!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> b(i);
        }
        bin.close();
    }
    else
    {
        // cout<<"Use All 1 as Vector B"<<endl;
        for(int i=0;i<M;i++){
            b(i)=1;
        }
    }

    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }
    
    clock_t  time_stt;

    //QR分解
    time_stt = clock();
    SparseQR<SparseMatrix<double>,AMDOrdering<int>> solver;
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
    // fout<<"QRSolver for " << A << " and " << B << " Solving Succeed!"<<endl;
    // fout<<"Compute time: "<<compute_time<<" ms"<<endl;
    // fout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    // fout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;

    // 计算残差向量的范数
    VectorXd residual = a*x-b;
    double residualNorm = residual.norm();
    double l1Norm = residual.lpNorm<1>();
    double infinityNorm = residual.lpNorm<Eigen::Infinity>();

    // fout<< "l1Norm norm: " << l1Norm << endl;
    // fout<< "Euclidean norm: " << residualNorm << endl;
    // fout<< "infinityNorm norm: " << infinityNorm << endl;

    fout<< std::setprecision(15) <<x<<endl;

    cout<<"QRSolver for " << A << " Solving Succeed!"<<endl;
    cout<<"Compute time: "<<compute_time<<" ms"<<endl;
    cout<<"Solve time: "<<solve_time<<" ms"<<endl<<endl;
    cout<<"Total time: "<<compute_time+solve_time<<" ms"<<endl<<endl;
    cout<< "Ax-b (1-norm): " << l1Norm << endl;
    cout<< "Ax-b (2-norm): " << residualNorm << endl;
    cout<< "Ax-b (infinite-norm): " << infinityNorm << endl;

    fout.close();
    return;
}
    

    
    int main(int argc, char** argv)
    {
        // if(argc < 4)
        // {
        //     return 0;
        // }
        if(argc < 2 || argc>4)
        {
            help_message();
            return 0;
        }
        else if(argc ==2)
        {
            BICGSolve(argv[1],NULL,NULL);
        }
        else if(argc==3)
        {
            BICGSolve(argv[1],argv[2],NULL);
        }
        else if(argc==4)
        {
            BICGSolve(argv[1],argv[2],argv[3]);
        }
        return 0;
    }
    
}
//g++ eigen.cpp -o eigen -I../include/Eigen -O2 -std=c++11