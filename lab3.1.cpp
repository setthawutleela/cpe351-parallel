#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

double** readFile(double** pMat, const char* fileName, int* nrows, int* ncols);
double** createMatrix(double** pMat, int nrows, int ncols);
double** calAddition(double** pMatRes, double** pMat1, double** pMat2, int startRow, int endRow, int ncols);
void printMatrix(double** pMat, int nrows, int ncols);
void freeData(double** pData, int ncols);
void writeFile(double** pMatRes, const char* fileName, int nrows, int ncols);

int main(int argc, char* argv[]) {
    int p;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int nrows, ncols, nrows1, ncols1, nrows2, ncols2, groupRows, remainRows;
        double** pMat1 = NULL;
        double** pMat2 = NULL;
        double** pMatRes = NULL;
        double startTime, endTime;

        pMat1 = readFile(pMat1, "input1.txt", &nrows1, &ncols1);
        pMat2 = readFile(pMat2, "input2.txt", &nrows2, &ncols2);

        if ((nrows1 != nrows2) || (ncols1 != ncols2)) {
            printf("Error! the size of both matrices are not the same\n");
            exit(0);
        }

        nrows = nrows1;
        ncols = ncols1;

        /*printf("\nMATRIX 1:\n\n");
        printMatrix(pMat1, nrows, ncols);
        printf("\nMATRIX 2:\n\n");
        printMatrix(pMat2, nrows, ncols);
        printf("\n");*/

        pMatRes = createMatrix(pMatRes, nrows, ncols);

        groupRows = nrows / p;
        remainRows = nrows - (groupRows * p);

        printf("nrows: %d | ncols: %d | p: %d | groupRows: %d | remainRows: %d\n\n", nrows, ncols, p, groupRows, remainRows);

        startTime = MPI_Wtime();

        if (p > 1) { // The amount of process is more than 1
            for (int i = 1; i < p; i++) {
                MPI_Send(&groupRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&ncols, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                for (int j = groupRows * i; j < (groupRows * i) + groupRows; j++) {
                    for (int k = 0; k < ncols; k++) {
                        MPI_Send(&pMat1[j][k], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                        MPI_Send(&pMat2[j][k], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                    }
                }
                printf("Rank 0 sent the information to rank %d succesfully\n", i);
            }
        }

        pMatRes = calAddition(pMatRes, pMat1, pMat2, 0, groupRows, ncols);

        if (remainRows > 1) { // Have the remaining rows after sending the tasks
            pMatRes = calAddition(pMatRes, pMat1, pMat2, groupRows * p, (groupRows * p) + remainRows, ncols);
        }

        for (int i = 1; i < p; i++) {
            for (int j = groupRows * i; j < (groupRows * i) + groupRows; j++) {
                for (int k = 0; k < ncols; k++) {
                    MPI_Recv(&pMatRes[j][k], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }

        endTime = MPI_Wtime();

        /*printf("\nRESULT:\n\n", rank);
        printMatrix(pMatRes, nrows, ncols);
        printf("\n");*/
        printf("Rank 0 Completed.\n");
        printf(">>>>>>>>>> TIME: %f sec <<<<<<<<<< \n\n", endTime - startTime);

        writeFile(pMatRes, "output.txt", nrows, ncols);

        freeData(pMat1, ncols);
        freeData(pMat2, ncols);
        freeData(pMatRes, ncols);
    }

    else {  // Other ranks
        int nrows, ncols;
        double** pMat1 = NULL;
        double** pMat2 = NULL;
        double** pMatRes = NULL;

        MPI_Recv(&nrows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&ncols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pMat1 = createMatrix(pMat1, nrows, ncols);
        pMat2 = createMatrix(pMat2, nrows, ncols);
        pMatRes = createMatrix(pMatRes, nrows, ncols);

        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                MPI_Recv(&pMat1[i][j], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&pMat2[i][j], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pMatRes[i][j] = pMat1[i][j] + pMat2[i][j];
            }
        }

        printf("Rank %d received the information from rank 0 successfully\n", rank);

        pMatRes = calAddition(pMatRes, pMat1, pMat2, 0, nrows, ncols);

        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                MPI_Send(&pMatRes[i][j], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }

        printf("Rank %d sent the result to rank 0\n", rank);
        printf("Rank %d completed\n", rank);
        freeData(pMat1, ncols);
        freeData(pMat2, ncols);
        freeData(pMatRes, ncols);
    }

    MPI_Finalize();
    return 0;
}

double** readFile(double** pMat, const char* fileName, int* nrows, int* ncols) {
    FILE* fp = NULL;
    int i = 0;          //for loop
    int j = 0;          //for loop
    char buffer[SHORTLEN] = { 0 };
    fp = fopen(fileName, "r");

    if (fp == NULL) {
        printf("Error! opening %s file\n", fileName);
        exit(0);
    }

    if (fgets(buffer, SHORTLEN, fp) != NULL) {
        sscanf(buffer, "%d %d", &*nrows, &*ncols);
        pMat = createMatrix(pMat, *nrows, *ncols);
    }

    while (fscanf(fp, "%s", buffer) == 1) {
        if (j >= *ncols) {
            j = 0;
            i++;
        }
        sscanf(buffer, "%lf", &pMat[i][j]);
        j++;
    }

    fclose(fp);
    return pMat;
}

double** createMatrix(double** pMat, int nrows, int ncols) {
    pMat = (double**)calloc(nrows, sizeof(double*));
    if (pMat == NULL) {
        printf("Error! allocating memory fail\n");
        exit(0);
    }
    for (int i = 0; i < nrows; i++) {
        pMat[i] = (double*)calloc(ncols, sizeof(double));
        if (pMat[i] == NULL) {
            printf("Error! allocating memory fail\n");
            exit(0);
        }
    }
    return pMat;
}

double** calAddition(double** pMatRes, double** pMat1, double** pMat2, int startRow, int endRow, int ncols) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < ncols; j++) {
            pMatRes[i][j] = pMat1[i][j] + pMat2[i][j];
        }
    }
    return pMatRes;
}

void printMatrix(double** pMat, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            printf("%.1lf  ", pMat[i][j]);
        }
        printf("\n");
    }
}

void freeData(double** pData, int ncols) {
    free(*pData);
    free(pData);
}

void writeFile(double** pMatRes, const char* fileName, int nrows, int ncols) {
    FILE* fp = NULL;
    fp = fopen(fileName, "w");

    if (fp == NULL) {
        printf("Error! opening %s file\n", fileName);
        exit(0);
    }

    fprintf(fp, "%d %d\n", nrows, ncols);

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fprintf(fp, "%.1f", pMatRes[i][j]);
            if (j < ncols - 1)
                fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Write file successfully\n");

}