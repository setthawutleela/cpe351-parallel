#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

void writeFile(double* pMatRes, const char* fileName, int nrows, int ncols) {
    FILE* fp = fopen(fileName, "w");
    fprintf(fp, "%d %d\n", nrows, ncols);
    int i = 0;
    while (i < nrows * ncols) {
        if (i % ncols == (ncols - 1))
            fprintf(fp, "%.10lf\n", pMatRes[i++]);
        else
            fprintf(fp, "%.10lf ", pMatRes[i++]);
    }
    fclose(fp);
}

int main(int argc, char* argv[]) {
    int p;
    int rank;
    int nrows1, ncols1, nrows2, ncols2, groupRows, remainRows;
    double* pMat1 = NULL, * pMat2 = NULL, * pMatRes = NULL;
    double* pMatBuf1 = NULL, * pMatBufRes = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (p > 1) {
        if (rank == 1) {
            FILE* fp = fopen(argv[2], "r");
            int i = 0;
            int j = 0;
            fscanf(fp, "%d %d", &nrows2, &ncols2);
            pMat2 = (double*)calloc(nrows2 * ncols2, sizeof(double));
            while (fscanf(fp, "%lf", &pMat2[(j * nrows2) + i]) == 1) {
                j++;
                if (j >= ncols2) {
                    j = 0;
                    i++;
                }
            }
            fclose(fp);
        }

        MPI_Bcast(&nrows2, 1, MPI_INT, 1, MPI_COMM_WORLD);
        MPI_Bcast(&ncols2, 1, MPI_INT, 1, MPI_COMM_WORLD);
        if (rank != 1)
            pMat2 = (double*)calloc(nrows2 * ncols2, sizeof(double));
        MPI_Bcast(&pMat2[0], nrows2 * ncols2, MPI_DOUBLE, 1, MPI_COMM_WORLD);

        if (rank == 0) {
            FILE* fp = fopen(argv[1], "r");;
            int i = 0;
            fscanf(fp, "%d %d", &nrows1, &ncols1);
            pMat1 = (double*)calloc(nrows1 * ncols1, sizeof(double));
            while (fscanf(fp, "%lf", &pMat1[i++]) == 1);
            fclose(fp);
            pMatRes = (double*)calloc(nrows1 * ncols2, sizeof(double));
        }

        MPI_Bcast(&nrows1, 1, MPI_INT, 0, MPI_COMM_WORLD);

        pMatBuf1 = (double*)calloc((nrows1 / p) * nrows2, sizeof(double));
        pMatBufRes = (double*)calloc((nrows1 / p) * ncols2, sizeof(double));
        MPI_Scatter(pMat1, (nrows1 / p) * nrows2, MPI_DOUBLE, pMatBuf1, (nrows1 / p) * nrows2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && (nrows1 - ((nrows1 / p) * p)) >= 1) {
            for (int i = (nrows1 / p) * p; i < nrows1; i++) {
                for (int j = 0; j < ncols2; j++) {
                    for (int k = 0; k < nrows2; k++) {
                        pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                    }
                }
            }
        }

        for (int i = 0; i < (nrows1 / p); i++) {
            for (int j = 0; j < ncols2; j++) {
                for (int k = 0; k < nrows2; k++) {
                    pMatBufRes[(ncols2 * i) + j] += pMatBuf1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                }
            }
        }
        MPI_Gather(pMatBufRes, (nrows1 / p) * ncols2, MPI_DOUBLE, pMatRes, (nrows1 / p) * ncols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if(rank == 0)
            writeFile(pMatRes, argv[3], nrows1, ncols2);
    }
    else {
        FILE* fp = fopen(argv[1], "r");
        int i = 0;
        int j = 0;
        fscanf(fp, "%d %d", &nrows1, &ncols1);
        pMat1 = (double*)calloc(nrows1 * ncols1, sizeof(double));
        while (fscanf(fp, "%lf", &pMat1[i++]) == 1);
        fclose(fp);
        i = 0;
        fp = fopen(argv[2], "r");
        fscanf(fp, "%d %d", &nrows2, &ncols2);
        pMat2 = (double*)calloc(nrows2 * ncols2, sizeof(double));
        while (fscanf(fp, "%lf", &pMat2[(j * nrows2) + i]) == 1) {
            j++;
            if (j >= ncols2) {
                j = 0;
                i++;
            }
        }
        fclose(fp);
        pMatRes = (double*)calloc(nrows1 * ncols2, sizeof(double));
        for (int i = 0; i < nrows1; i++) {
            for (int j = 0; j < ncols2; j++) {
                for (int k = 0; k < nrows2; k++) {
                    pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                }
            }
        }
        writeFile(pMatRes, argv[3], nrows1, ncols2);
    }
    MPI_Finalize();
    return 0;
}