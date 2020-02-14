#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define SHORTLEN 32

#pragma warning (disable : 4996)

float** readFile(const char* fileName, int* nrows, int* ncols);
float** createMatrix(int nrows, int ncols);
float** calAddition(float** pMatRes, float** pMat1, float** pMat2, int startRow, int endRow, int ncols);
void printMatrix(float** pMat, int nrows, int ncols);
void freeData(float*** pData, int ncols);
void writeFile(float** pMatRes, const char* fileName, int nrows, int ncols);

int main(int argc, char* argv[]) {
    int p;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request request[5];

    if (rank == 0) {
        int nrows, ncols, nrows1, ncols1, nrows2, ncols2, groupRows, remainRows;
        float** pMat1 = NULL;
        float** pMat2 = NULL;
        float** pMatRes = NULL;
        double startTime, endTime;

        pMat1 = readFile("matAlarge.txt", &nrows1, &ncols1);
        pMat2 = readFile("matBlarge.txt", &nrows2, &ncols2);

        nrows = nrows1;
        ncols = ncols1;

        pMatRes = createMatrix(nrows, ncols);

        groupRows = nrows / p;
        remainRows = nrows - (groupRows * p);

        printf("nrows: %d | ncols: %d | p: %d | groupRows: %d | remainRows: %d\n\n", nrows, ncols, p, groupRows, remainRows);

        startTime = MPI_Wtime();

        if (p > 1) { // The amount of process is more than 1

            for (int i = 1; i < p; i++) {
                //MPI_Send(&groupRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                //MPI_Send(&ncols, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
                MPI_Isend(&groupRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request[0]);
                MPI_Isend(&ncols, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &request[1]);
            }

            if (remainRows >= 1) { // Have the remaining rows after sending the tasks
                pMatRes = calAddition(pMatRes, pMat1, pMat2, groupRows * p, (groupRows * p) + remainRows, ncols);
            }

            for (int i = 1; i < p; i++) {
                //MPI_Send(&(pMat1[groupRows * i][0]), groupRows * ncols, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
                //MPI_Send(&(pMat2[groupRows * i][0]), groupRows * ncols, MPI_FLOAT, i, 3, MPI_COMM_WORLD);
                MPI_Isend(&(pMat1[groupRows * i][0]), groupRows * ncols, MPI_FLOAT, i, 2, MPI_COMM_WORLD, &request[2]);
                MPI_Isend(&(pMat2[groupRows * i][0]), groupRows * ncols, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &request[3]);
                printf("Rank 0 sent the information to rank %d succesfully\n", i);
            }

            for (int i = 1; i < p; i++) {
                //MPI_Recv(&(pMatRes[groupRows * i][0]), groupRows * ncols, MPI_FLOAT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Irecv(&pMatRes[groupRows * i][0], groupRows * ncols, MPI_FLOAT, i, 4, MPI_COMM_WORLD, &request[4]);
                MPI_Wait(&request[4], MPI_STATUS_IGNORE);
            }
        }

        pMatRes = calAddition(pMatRes, pMat1, pMat2, 0, groupRows, ncols);

        endTime = MPI_Wtime();

        printf("Rank 0 completed.\n");
        printf("\t====================================\n");
        printf("\t|| Calculation time: %f sec ||\n", endTime - startTime);
        printf("\t====================================\n");
        writeFile(pMatRes, "output.txt", nrows, ncols);
        freeData(&pMat1, ncols);
        freeData(&pMat2, ncols);
        freeData(&pMatRes, ncols);
    }

    else {  // Other ranks
        int nrows, ncols;
        float** pMat1 = NULL;
        float** pMat2 = NULL;
        float** pMatRes = NULL;

        //MPI_Recv(&nrows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(&ncols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Irecv(&nrows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&ncols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &request[1]);
        MPI_Wait(&request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&request[1], MPI_STATUS_IGNORE);
        pMat1 = createMatrix(nrows, ncols);
        pMat2 = createMatrix(nrows, ncols);
        pMatRes = createMatrix(nrows, ncols);

        //MPI_Recv(&(pMat1[0][0]), nrows * ncols, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //MPI_Recv(&(pMat2[0][0]), nrows * ncols, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Irecv(&pMat1[0][0], nrows * ncols, MPI_FLOAT, 0, 2, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(&pMat2[0][0], nrows * ncols, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &request[3]);
        MPI_Wait(&request[2], MPI_STATUS_IGNORE);
        MPI_Wait(&request[3], MPI_STATUS_IGNORE);
        printf("Rank %d received the information from rank 0 successfully\n", rank);

        pMatRes = calAddition(pMatRes, pMat1, pMat2, 0, nrows, ncols);
        //MPI_Send(&(pMatRes[0][0]), nrows * ncols, MPI_FLOAT, 0, 4, MPI_COMM_WORLD);
        MPI_Isend(&pMatRes[0][0], nrows * ncols, MPI_FLOAT, 0, 4, MPI_COMM_WORLD, &request[4]);

        printf("Rank %d sent the result to rank 0\n", rank);
        printf("Rank %d completed\n", rank);
        freeData(&pMat1, ncols);
        freeData(&pMat2, ncols);
        freeData(&pMatRes, ncols);
    }

    MPI_Finalize();
    return 0;
}

float** readFile(const char* fileName, int* nrows, int* ncols) {
    FILE* fp = NULL;
    float** pData = NULL;
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
        pData = createMatrix(*nrows, *ncols);
    }

    while (fscanf(fp, "%s", buffer) == 1) {
        if (j >= *ncols) {
            j = 0;
            i++;
        }
        sscanf(buffer, "%f", &pData[i][j]);
        j++;
    }

    fclose(fp);
    return pData;
}

float** createMatrix(int nrows, int ncols) {
    float* data = (float*)calloc(nrows * ncols, sizeof(float));
    if (data == NULL) {
        printf("Error! cannot allocate memory\n");
        exit(0);
    }
    float** array = (float**)calloc(nrows, sizeof(float*));
    if (array == NULL) {
        printf("Error! cannot allocate memory\n");
        exit(0);
    }
    for (int i = 0; i < nrows; i++) {
        array[i] = &data[ncols * i];
    }
    return array;
}

float** calAddition(float** pMatRes, float** pMat1, float** pMat2, int startRow, int endRow, int ncols) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < ncols; j++) {
            pMatRes[i][j] = pMat1[i][j] + pMat2[i][j];
        }
    }
    return pMatRes;
}

void printMatrix(float** pMat, int nrows, int ncols) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            printf("%.2lf  ", pMat[i][j]);
        }
        printf("\n");
    }
}

void freeData(float*** pData, int ncols) {
    free(**pData);
    free(*pData);
}

void writeFile(float** pMatRes, const char* fileName, int nrows, int ncols) {
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