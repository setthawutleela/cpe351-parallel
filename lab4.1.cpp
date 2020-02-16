#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

int* readFile(const char* fileName, int* nrows, int* ncols, int isReverse);
int* createMatrix(int nrows, int ncols);
int* multipleMatrix(int* pMatRes, int* pMat1, int* pMat2, int nrows1, int ncols1, int nrows2, int ncols2, int nrowsRes, int ncolsRes);
void printMatrix(int* pMat, int nrows, int ncols, int isReverse);
void print1DArray(int* pMat, int size);
void writeFile(int* pMatRes, const char* fileName, int nrows, int ncols);

int main(int argc, char* argv[]) {
    int p;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int nrows1, ncols1, nrows2, ncols2, nrowsRes, ncolsRes;
        int* pMat1 = NULL;
        int* pMat2 = NULL;
        int* pMatRes = NULL;
        double startTime, endTime;

        pMat1 = readFile("matAsmall.txt", &nrows1, &ncols1, 0);
        printf("row1: %d | col1: %d\n", nrows1, ncols1);
        pMat2 = readFile("matBsmall.txt", &nrows2, &ncols2, 1);
        printf("row2: %d | col2: %d\n", nrows2, ncols2);

        /*printf("MATRIX A:\n");
        print1DArray(pMat1, nrows1 * ncols1);
        printMatrix(pMat1, nrows1, ncols1, 0);
        printf("\n");
        printf("MATRIX B:\n");
        print1DArray(pMat2, nrows2 * ncols2);
        printMatrix(pMat2, nrows2, ncols2, 1);*/

        // The number of rows in matrix 1 isn't equal to the number of columns in matrix 2
        if (ncols1 != nrows2) {
            printf("Error! cannot multiply both of them due to the number of rows in matrix 1 isn't equal to the number of columns in matrix 2\n");
            MPI_Finalize();
            exit(0);
        }

        nrowsRes = nrows1;
        ncolsRes = ncols2;

        pMatRes = createMatrix(nrowsRes, ncolsRes);

        startTime = MPI_Wtime();
        pMatRes = multipleMatrix(pMatRes, pMat1, pMat2, nrows1, ncols1, nrows2, ncols2, nrowsRes, ncolsRes);
        endTime = MPI_Wtime();

        /*printf("\n");
        printf("MATRIX RESULT:\n");
        print1DArray(pMatRes, nrowsRes * ncolsRes);
        printMatrix(pMatRes, nrowsRes, ncolsRes, 0);*/

        printf("Rank 0 completed.\n");
        printf("\t====================================\n"); 
        printf("\t|| Calculation time: %f sec ||\n", endTime - startTime);
        printf("\t====================================\n");
        //writeFile(pMatRes, "output.txt", nrowsRes, ncolsRes);
        free(pMat1);
        free(pMat2);
        free(pMatRes);
    }

    else {  // Other ranks
        /*int nrows, ncols;
        int* pMat1 = NULL;
        int* pMat2 = NULL;
        int* pMatRes = NULL;*/

        /***************************
        *      Coding here
        ****************************/

        /*printf("Rank %d sent the result to rank 0\n", rank);
        printf("Rank %d completed\n", rank);*/
    }

    MPI_Finalize();
    return 0;
}

int* readFile(const char* fileName, int* nrows, int* ncols, int isReverse) {
    FILE* fp = NULL;
    int* pData = NULL;
    int i = 0;
    int j = 0;
    char buffer[SHORTLEN] = { 0 };
    fp = fopen(fileName, "r");

    if (fp == NULL) {
        printf("Error! opening %s file\n", fileName);
        exit(0);
    }

    if (fgets(buffer, SHORTLEN, fp) != NULL) {
        sscanf(buffer, "%d %d", &*nrows, &*ncols);
        pData = createMatrix(*nrows, *ncols);
    }

    if (isReverse == 0) {   // First matrix
        while (fscanf(fp, "%s", buffer) == 1) {
            sscanf(buffer, "%d", &pData[i]);
            i++;
        }
    }
    else {  // Second matrix
        while (fscanf(fp, "%s", buffer) == 1) {
            if (j >= *ncols) {
                j = 0;
                i++;
            }
            sscanf(buffer, "%d", &pData[(j*(*nrows))+i]);
            j++;
        }
    }

    fclose(fp);
    return pData;
}

int* createMatrix(int nrows, int ncols) {
    int* pData = (int*)calloc(nrows * ncols, sizeof(int));
    if (pData == NULL) {
        printf("Error! cannot allocate memory");
        exit(0);
    }
    return pData;
}

int* multipleMatrix(int* pMatRes, int* pMat1, int* pMat2, int nrows1, int ncols1, int nrows2, int ncols2, int nrowsRes, int ncolsRes) {
    for (int i = 0; i < nrows1; i++) {
        for (int j = 0; j < ncols2; j++) {
            for (int k = 0; k < nrows2; k++) {
                //int posRes = (ncols2 * i) + j
                //int pos1 = (nrows2 * i) + k;
                //int pos2 = (nrows2 * j) + k;
                //printf("%d-%d  ", pos1, pos2);
                pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
            }
            //printf("\n");
        }
    }
    return pMatRes;
}

void printMatrix(int* pMat, int nrows, int ncols, int isReverse) {
    if (isReverse == 0) {
        for (int i = 0; i < nrows; i++) {
            printf("\t");
            for (int j = 0; j < ncols; j++) {
                printf("%4d  ", pMat[(i * ncols) + j]);
            }
            printf("\n");
        }
    }
    else {
        for (int i = 0; i < nrows; i++) {
            printf("\t");
            for (int j = 0; j < ncols; j++) {
                printf("%4d  ", pMat[(j * nrows) + i]);
            }
            printf("\n");
        }
    }
}

void print1DArray(int* pMat, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", pMat[i]);
    }
    printf("\n");
}