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
#define MAX_N 5000	//???��??????????
#define MAXREPT 10000
#define epsilon 0.01   //?????

void Gauss_Seidel(char *A, char *B, char*X)
{
    int MAX_N = 5000;
    int MAXREPT = 10000;
    double epsilon = 0.01;
    int i, j, k;
    // ��ȡ.mtx�ļ�
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
    ofstream fout(X, ios::out | ios::trunc); //���ļ�������ʱ�������ļ��������ļ��Ѵ���ʱ���ԭ�����ݲ�д��������
    if (!fout)
    {
        cout << "File " << X << " Open Failed!" << endl;
        return;
    }
    int rows, cols, nnz;
    while (ain.peek() == '%')
        ain.ignore(2048, '\n');
    ain >> rows >> cols >> nnz;

    // ����ϡ���������ݽṹ
    std::vector<std::vector<double>> sparse_matrix(rows, std::vector<double>(cols, 0.0));

    // ��ȡ����Ԫ�ز�������
    for (i = 0; i < nnz; ++i) {
        int row, col;
        double value;
        ain >> row >> col >> value;

        // ������Ԫ���������
        sparse_matrix[row - 1][col - 1] = value; // ��һ����Ϊ .mtx �ļ�ʹ�� 1-based ����
    }

    // ��ϡ�����ת��Ϊ���飨��ά���飩
    std::vector<std::vector<double>> a = sparse_matrix;
	double n=rows;
	double err,s;
	static double b[MAX_N][MAX_N], c[MAX_N], g[MAX_N];
	static double x[MAX_N], nx[MAX_N];
	
	//��ʼ��������

	//�����������C����
	
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

	//������������  x^(k+1) = b * x^(k)+ g  ��������B �� g [ x^k��ʾ��k�λ�õ�x]
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (a[i][i] == 0)
				cerr << "��ʱ��֧��ϵ������Խ���Ԫ��Ϊ0" << endl;
			b[i][j] = -a[i][j] / a[i][i];
			g[i] = c[i] / a[i][i];			//�ȼ���a[i][i]!=0,ѧϰ����ʱ�ò���
		}
		b[i][i] = 0;
	}
	//��ʼ��nx[i]=0
	memset(nx,0,sizeof(double)*n);

	for (i = 0; i < MAXREPT; i++) {//������������
        
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

		err = 0;			//x^(k+1)-x^k�����
		for (j = 0; j < n; j++) {
			if (err < fabs(nx[j] - x[j])) {
				err = fabs(nx[j] - x[j]);		//��������ʽ
			}
		}
	
		if (err < epsilon) {		//�������
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
?????�
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
