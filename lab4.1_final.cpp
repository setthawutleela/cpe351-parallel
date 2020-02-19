#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

void writeFile(double* pMatRes, const char* fileName, int nrows, int ncols);

int main(int argc, char* argv[]) {
    int p;
    int rank;
    int nrows1, ncols1, nrows2, ncols2, groupRows, remainRows;
    double* pMat1 = NULL, * pMat2 = NULL, * pMatRes = NULL;
    double* pMatBuf1 = NULL, * pMatBufRes = NULL;
    double startTime, endTime;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (p > 1) {
        if (rank == 0) {
            FILE* fp = NULL;
            int i = 0;
            int j = 0;
            char buffer[SHORTLEN] = { 0 };

            fp = fopen(argv[1], "r");
            fgets(buffer, SHORTLEN, fp);
            sscanf(buffer, "%d %d", &nrows1, &ncols1);
            pMat1 = (double*)calloc(nrows1 * ncols1, sizeof(double));
            while (fscanf(fp, "%lf", &pMat1[i]) == 1) {
                i++;
            }
            fclose(fp);
        
            i = 0;
            j = 0;

            fp = fopen(argv[2], "r");
            fgets(buffer, SHORTLEN, fp);
            sscanf(buffer, "%d %d", &nrows2, &ncols2);
            pMat2 = (double*)calloc(nrows2 * ncols2, sizeof(double));

            while (fscanf(fp, "%lf", &pMat2[(j * nrows2) + i]) == 1) {
                j++;
                if (j >= ncols2) {
                    j = 0;  
                    i++;
                }
            }
            fclose(fp);

            groupRows = nrows1 / p;
            remainRows = nrows1 - (groupRows * p);
            pMatRes = (double*)calloc(nrows1 * ncols2, sizeof(double));
        }

        MPI_Bcast(&nrows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ncols1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nrows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ncols2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&groupRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            pMat2 = (double*)calloc(nrows2 * ncols2, sizeof(double));
            pMatBuf1 = (double*)calloc(groupRows * ncols1, sizeof(double));
            pMatBufRes = (double*)calloc(groupRows * ncols2, sizeof(double));
        }

        MPI_Bcast(&pMat2[0], nrows2 * ncols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (p > 1) {
                for (int i = 1; i < p; i++)
                    MPI_Send(&(pMat1[groupRows * ncols1 * i]), groupRows * ncols1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
            if (remainRows >= 1) {
                for (int i = groupRows * p; i < nrows1; i++) {
                    for (int j = 0; j < ncols2; j++) {
                        for (int k = 0; k < nrows2; k++)
                            pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                    }
                }
            }
            for (int i = 0; i < groupRows; i++) {
                for (int j = 0; j < ncols2; j++) {
                    for (int k = 0; k < nrows2; k++)
                        pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                }
            }
            if (p > 1) {
                for (int i = 1; i < p; i++)
                    MPI_Recv(&(pMatRes[groupRows * ncols2 * i]), groupRows * ncols2, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            writeFile(pMatRes, argv[3], nrows1, ncols2);
        }

        else {
            MPI_Recv(&(pMatBuf1[0]), groupRows * ncols1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < groupRows; i++) {
                for (int j = 0; j < ncols2; j++) {
                    for (int k = 0; k < nrows2; k++) {
                        pMatBufRes[(ncols2 * i) + j] += pMatBuf1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
                    }
                }
            }
            MPI_Send(&(pMatBufRes[0]), groupRows * ncols2, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
        }
    }
    else {
        FILE* fp = NULL;
        int i = 0;
        int j = 0;
        char buffer[SHORTLEN] = { 0 };

        fp = fopen(argv[1], "r");
        fgets(buffer, SHORTLEN, fp);
        sscanf(buffer, "%d %d", &nrows1, &ncols1);
        pMat1 = (double*)calloc(nrows1 * ncols1, sizeof(double));
        while (fscanf(fp, "%lf", &pMat1[i]) == 1) {
            i++;
        }
        fclose(fp);

        i = 0;
        j = 0;

        fp = fopen(argv[2], "r");
        fgets(buffer, SHORTLEN, fp);
        sscanf(buffer, "%d %d", &nrows2, &ncols2);
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

void writeFile(double* pMatRes, const char* fileName, int nrows, int ncols) {
    FILE* fp = NULL;
    int i = 0;
    fp = fopen(fileName, "w");
    fprintf(fp, "%d %d\n", nrows, ncols);
    while (i < nrows * ncols) {
        fprintf(fp, "%.10lf", pMatRes[i]);
        if (i % ncols == ncols - 1)
            fprintf(fp, "\n");
        else
            fprintf(fp," ");
        i++;
    }
    fclose(fp);
}