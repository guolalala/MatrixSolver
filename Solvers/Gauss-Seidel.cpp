//
//	Gauss-Seidel	  ????????????????		//
//
#include<math.h>
#include<iostream>
#include<string.h>
#include <fstream>
#include <vector>
#include <sstream>
using namespace std;
#define MAX_N 5000	//???÷??????????
#define MAXREPT 10000
#define epsilon 0.01   //?????

void Gauss_Seidel(char *A, char *B, char*X)
{
    int MAX_N = 5000;
    int MAXREPT = 10000;
    double epsilon = 0.01;
    int i, j, k;
    // 读取.mtx文件
    ifstream ain(A);
    if(!ain)
    {
        cout<<"File A Read Failed!"<<endl;
        return 0;
    }
    const char * defaultName = "output_x.dat";
    if(X==NULL)
    {
        X = (char *)malloc(sizeof(char)*105);
        strcpy(X,defaultName);
    }
    ofstream fout(X, ios::out | ios::trunc); //在文件不存在时创建新文件，并在文件已存在时清除原有数据并写入新数据
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }
    int rows, cols, nnz;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> rows >> cols >> nnz;

    // 构造稀疏矩阵的数据结构
    std::vector<std::vector<double>> sparse_matrix(rows, std::vector<double>(cols, 0.0));

    // 读取非零元素并填充矩阵
    for (i = 0; i < nnz; ++i) {
        int row, col;
        double value;
        ain >> row >> col >> value;

        // 将非零元素填入矩阵
        sparse_matrix[row - 1][col - 1] = value; // 减一是因为 .mtx 文件使用 1-based 索引
    }

    // 将稀疏矩阵转换为数组（二维数组）
    std::vector<std::vector<double>> a = sparse_matrix;
	double n=rows;
	double err,s;
	static double b[MAX_N][MAX_N], c[MAX_N], g[MAX_N];
	static double x[MAX_N], nx[MAX_N];
	
	//初始化工作：

	//任务二：输入C矩阵
	
    if(B!=NULL)
    {
        ifstream bin(B);
        if(!bin)
        {
            cout << "File " << B <<" Read Failed!" << endl;
            return 0;
        }
        while (bin.peek() == '%')
            bin.ignore(2048, '\n');
        bin >> M >> N;
        for (int i = 0; i < M; ++i) {
            bin >> c[i];
        }
        bin.close();
    }
    else
    {
        for (i = 0; i < n; i++)  c[i]=1;
    }

    clock_t  time_stt;
    time_stt = clock();

	//任务三：生成  x^(k+1) = b * x^(k)+ g  迭代矩阵B 和 g [ x^k表示第k次获得的x]
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (a[i][i] == 0)
				cerr << "暂时不支持系数矩阵对角线元素为0" << endl;
			b[i][j] = -a[i][j] / a[i][i];
			g[i] = c[i] / a[i][i];			//先假设a[i][i]!=0,学习中暂时用不到
		}
		b[i][i] = 0;
	}
	//初始化nx[i]=0
	memset(nx,0,sizeof(double)*n);

	for (i = 0; i < MAXREPT; i++) {//最大迭代周期数
        
        if(i%100==0)
            cout<<"Iteration "<<i<<endl;

		for (j = 0; j < n; j++)
			x[j] = nx[j];

		for (j = 0; j < n; j++) {
			s = g[j];
			for (k = 0; k < n; k++) {
				if (j == k) continue;
				s += b[j][k] * x[k];		
			}
			nx[j] = s;
		}

		err = 0;			//x^(k+1)-x^k的误差
		for (j = 0; j < n; j++) {
			if (err < fabs(nx[j] - x[j])) {
				err = fabs(nx[j] - x[j]);		//求最大差即最大范式
			}
		}
	
		if (err < epsilon) {		//控制误差
			cout<<"BiCGSTAB for " << A <<  " Solving Succeed!"<<endl;
            double compute_time = 1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC;
            cout<<"Total time: "<<compute_time<<" ms"<<endl<<endl;
            fout<< std::setprecision(15) <<x<<endl;
			// for (j = 0; j < n; j++)
			// 	cout << x[j] << endl;
			return 0;
		}
	}
	printf("After %d repeat ,no result\n", MAXREPT);
	return 1;
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

/*
?????飺
	64x1 - 3x2 - x3 =14
	2x1 -90 x2+x3 = -5
	x1 + x2 + 40x3 = 20
	??????????
	3
	64 -3 -1 2 -90 1 1 1 40
	14 -5 20
*/

//????
//Input n value(dim of AX = C) :3
//Now input the matrix a(i, j), i, j = 0, ..., 2 :
//	64 - 3 - 1 2 - 90 1 1 1 40
//	Now input the matrix b(i), i = 0, ..., 2
//	14 - 5 20
//	Solve ... x_i =
//	0.229547
//	0.0661302
//	0.492608
