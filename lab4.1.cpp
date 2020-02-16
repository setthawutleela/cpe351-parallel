#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

int* readFile(const char* fileName, int* nrows, int* ncols, int isReverse);
int* createMatrix(int nrows, int ncols);
int* multiplyMatrix(int* pMatRes, int* pMat1, int* pMat2, int startRow, int nrows1, int ncols1, int nrows2, int ncols2, int nrowsRes, int ncolsRes);
void printMatrix(int* pMat, int nrows, int ncols, int isReverse);
void print1DArray(int* pMat, int size);
void writeFile(int* pMatRes, const char* fileName, int nrows, int ncols);

int main(int argc, char* argv[]) {
    int p;
    int rank;
    int nrows1, ncols1, nrows2, ncols2, groupRows, remainRows;
    int* pMat1 = NULL, * pMat2 = NULL, * pMatRes = NULL;
    int* pMatBuf1 = NULL, * pMatBufRes = NULL;
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read file
    if (rank == 0) {
        pMat1 = readFile("matAsmall.txt", &nrows1, &ncols1, 0);
        pMat2 = readFile("matBsmall.txt", &nrows2, &ncols2, 1);

        // The number of rows in matrix 1 isn't equal to the number of columns in matrix 2
        if (ncols1 != nrows2) {
            printf("Error! cannot multiply both of them due to the number of rows in matrix 1 isn't equal to the number of columns in matrix 2\n");
            MPI_Finalize();
            exit(0);
        }

        groupRows = nrows1/p;   
        remainRows = nrows1 - (groupRows * p);

        printf("nrows1: %d | ncols1: %d | nrows2: %d | ncols2: %d | groupRows: %d | remainRows: %d\n", nrows1, ncols1, nrows2, ncols2, groupRows, remainRows);

        pMatRes = createMatrix(nrows1, ncols2);

    }

    // Bcast the information to all ranks
    if (p > 1) {
        MPI_Bcast(&groupRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nrows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ncols1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nrows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ncols2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        printf("Rank %d Bcast completely!\n", rank);
    }

    // Allocate memory in other ranks
    if(rank != 0) {
        //printf("RANK %d: \n", rank);
        //printf("row1: %d | col1: %d | row2: %d | col2: %d\n", nrows1, ncols1, nrows2, ncols2);
        pMatBuf1 = createMatrix(groupRows, ncols1);
        pMat2 = createMatrix(nrows2, ncols2);
        pMatBufRes = createMatrix(groupRows, ncols2);
    }

    if (p > 1) {
        MPI_Bcast(&pMat2[0], nrows2 * ncols2, MPI_INT, 0, MPI_COMM_WORLD);
        printf("Bcast pMat2 to other ranks in Rank %d\n", rank);
    }

    // Calculate in rank 0
    if (rank == 0) {
        startTime = MPI_Wtime();
        if (p > 1) {
            for (int i = 1; i < p; i++) {
                MPI_Send(&(pMat1[groupRows * ncols1 * i]), groupRows * ncols1, MPI_INT, i, 0, MPI_COMM_WORLD);
                printf("Rank %d send the information to rank %d successfully\n", rank, i);
            }

            if (remainRows >= 1) {
                pMatRes = multiplyMatrix(pMatRes, pMat1, pMat2, groupRows * p, nrows1, ncols1, nrows2, ncols2, nrows1, ncols2);
            }

            for (int i = 1; i < p; i++) {
                MPI_Recv(&(pMatRes[groupRows * ncols2 * i]), groupRows * ncols2, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("Rank %d received the result from rank %d\n", rank, i);
            }
        }

        pMatRes = multiplyMatrix(pMatRes, pMat1, pMat2, 0, groupRows, ncols1, nrows2, ncols2, nrows1, ncols2);

        endTime = MPI_Wtime();

        printf("Rank 0 completed.\n");
        printf("\t====================================\n");
        printf("\t|| Calculation time: %f sec ||\n", endTime - startTime);
        printf("\t====================================\n");

        writeFile(pMatRes, "output.txt", nrows1, ncols2);

        free(pMat1);
        free(pMat2);
        free(pMatRes);
    }

    // Calculate in other ranks
    else {
        MPI_Recv(&(pMatBuf1[0]), groupRows * ncols1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pMatBufRes = multiplyMatrix(pMatBufRes, pMatBuf1, pMat2, 0, groupRows, ncols1, nrows2, ncols2, groupRows, ncols2);
        MPI_Send(&(pMatBufRes[0]), groupRows * ncols2, MPI_INT, 0, 1, MPI_COMM_WORLD);

        printf("Rank %d sent the result to rank 0\n", rank);
        printf("Rank %d complete\n", rank);

        free(pMatBuf1);
        free(pMat2);
        free(pMatBufRes);
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

int* multiplyMatrix(int* pMatRes, int* pMat1, int* pMat2, int startRow, int nrows1, int ncols1, int nrows2, int ncols2, int nrowsRes, int ncolsRes) {
    for (int i = startRow; i < nrows1; i++) {
        for (int j = 0; j < ncols2; j++) {
            for (int k = 0; k < nrows2; k++) {
                //int posRes = (ncols2 * i) + j
                //int pos1 = (nrows2 * i) + k;
                //int pos2 = (nrows2 * j) + k;
                //printf("%d-%d  ", pos1, pos2);
                pMatRes[(ncols2 * i) + j] += pMat1[(nrows2 * i) + k] * pMat2[(nrows2 * j) + k];
            }
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

void writeFile(int* pMatRes, const char* fileName, int nrows, int ncols) {
    FILE* fp = NULL;
    fp = fopen(fileName, "w");

    if (fp == NULL) {
        printf("Error! opening %s file\n", fileName);
        exit(0);
    }

    fprintf(fp, "%d %d\n", nrows, ncols);

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fprintf(fp, "%d", pMatRes[(i * ncols) + j]);
            if (j < ncols - 1)
                fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Write file successfully\n");

}