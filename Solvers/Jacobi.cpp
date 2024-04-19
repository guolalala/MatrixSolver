//
//		Jacobi  迭代求解线性方程组		//
//
#include<math.h>
#define MAX_N 20		//设置方程的最大维数
#define MAXREPT 100
#define epsilon  0.00001	//求解精度
#include<iostream>
using namespace std;

int main()
{
	int n;
	int i, j, k;
	double err;
	static double a[MAX_N][MAX_N], b[MAX_N][MAX_N], c[MAX_N], g[MAX_N];
	static double x[MAX_N], nx[MAX_N];
	printf("\nInput n value(dim of AX=C):");//输入方程维度n*n
	cin>>n; 
	if (n > MAX_N) {
		cout << "The input n is larger than Max_N,please redefine the Max_N." << endl;
		return 1;
	}
	else if (n <= 0) {
		printf("Please input a number between 1 and %d\n", MAX_N);
		return 1;
	}
	//初始化工作：
	//任务一：开始输入AX=C的系数矩阵A
	printf("Now input the matrix a(i,j),i,j=0,...,%d:\n", n - 1);
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			cin >> a[i][j];
	//任务二：输入C矩阵
	printf("Now input the matrix b(i),i=0,...,%d\n", n - 1);
	for (i = 0; i < n; i++)  cin >> c[i];
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
	for (i = 0; i < MAXREPT; i++) {//最大迭代周期数

		for (j = 0; j < n; j++)
			nx[j] = g[j];
		for (j = 0; j < n; j++) {
			for (k = 0; k < n; k++) {
				if (j == k) continue;
				nx[j] += b[j][k] * x[k];		// x^(k+1) = b * x^(k)+ g  迭代
			}
		}

		err = 0;			//x^(k+1)-x^k的误差
		for (j = 0; j < n; j++) {
			if (err < fabs(nx[j] - x[j])) {
				err = fabs(nx[j] - x[j]);		//求最大差即最大范式
			}
		}

		for (j = 0; j < n; j++)  x[j] = nx[j];

		if (err < epsilon) {		//控制误差
			cout << "Solve ... x_i=" << endl;
			for (j = 0; j < n; j++)
				cout << x[j] << endl;
			return 0;
		}
	}
	printf("After %d repeat ,no result\n", MAXREPT);
	return 1;
}

/*
方程组：
	64x1 - 3x2 - x3 =14
	2x1 -90 x2+x3 = -5
	x1 + x2 + 40x3 = 20
	输入样例：
	3
	64 -3 -1 2 -90 1 1 1 40
	14 -5 20
*/
