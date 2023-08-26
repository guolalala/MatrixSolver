#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include "nicslu_cpp.inl"

extern "C" __declspec(dllexport) void NICSLUSolve(char* A, char* B, char* X);

const char *const ORDERING_METHODS[] = { "", "", "", "", "AMD", "AMM", "AMO1","AMO2","AMO3","AMDF" };

void NICSLUSolve(char* A, char* B, char* X)
{
    int ret;
    _double_t *ax = NULL, *b = NULL, *x = NULL;
    _uint_t *ai = NULL, *ap = NULL;
    _uint_t n, row, col, nz, nnz, i, j;
    CNicsLU solver;
    _double_t res[4], cond, det1, det2, fflop, sflop;
    size_t mem;
    FILE* fp = fopen(X, "w");

    //print license
    //PrintNicsLULicense(NULL);

    //read matrix A
    if (__FAIL(ReadMatrixMarketFile(A, &row, &col, &nz, NULL, NULL, NULL, NULL, NULL, NULL)))
    {
        printf("Failed to read matrix A\n");
        goto EXIT;
    }
    n = row;
    nnz = nz;
    ax = new _double_t[nnz];
    ai = new _uint_t[nnz];
    ap = new _uint_t[n + 1];
    ReadMatrixMarketFile(A, &row, &col, &nz, ax, ai, ap, NULL, NULL, NULL);
    //printf("Matrix A: row %d, col %d, nnz %d\n", n, n, nnz);

    //read RHS B
    b = new _double_t[n];
    ReadMatrixMarketFile(B, &row, &col, &nz, b, NULL, NULL, NULL, NULL, NULL);

    x = new _double_t[n];
    memset(x, 0, sizeof(_double_t) * n);

    //initialize solver
    ret = solver.Initialize();
    if (__FAIL(ret))
    {
        printf("Failed to initialize, return = %d\n", ret);
        goto EXIT;
    }
    //printf("NICSLU version %.0lf\n", solver.GetInformation(31));
    printf("NICSLUSolver for %s and %s Solving Succeed!\n", A, B);
    fprintf(fp,"NICSLUSolver for %s and %s Solving Succeed!\n", A, B);
    solver.SetConfiguration(0, 1.); //enable timer

    //pre-ordering (do only once)
    solver.Analyze(n, ax, ai, ap, MATRIX_ROW_REAL);
    printf("analysis time: %g\n", solver.GetInformation(0));
    fprintf(fp,"analysis time: %g\n", solver.GetInformation(0));
    printf("best ordering method: %s\n", ORDERING_METHODS[(int)solver.GetInformation(16)]);
    fprintf(fp,"best ordering method: %s\n", ORDERING_METHODS[(int)solver.GetInformation(16)]);

    //create threads (do only once)
    solver.CreateThreads(0); //use all physical cores

    //factor & solve (first-time)
    solver.FactorizeMatrix(ax, 0); //use all created threads
    printf("factor time: %g\n", solver.GetInformation(1));
    fprintf(fp,"factor time: %g\n", solver.GetInformation(1));
    solver.Solve(b, x);
    printf("solve time: %g\n", solver.GetInformation(2));
    fprintf(fp,"solve time: %g\n", solver.GetInformation(2));

    SparseResidual(n, ax, ai, ap, b, x, res, MATRIX_ROW_REAL);
    printf("residual RMSE: %g\n\n", res[0]);
    fprintf(fp,"residual RMSE: %g\n\n", res[0]);

    fprintf(fp, "x:");
    for (i = 0; i < n; ++i) {
        fprintf(fp,"%g\n", x[i]);
    }

EXIT:
    delete[]ax;
    delete[]ai;
    delete[]ap;
    delete[]b;
    delete[]x;
    solver.Free();
    fclose(fp);
    return;
}