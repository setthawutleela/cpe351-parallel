#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma warning (disable : 4996)

void writeFile(float* pMat, const char* fileName, int nrows, int ncols) {
    FILE* fp = fopen(fileName, "w");
    fprintf(fp, "%d %d\n", nrows, ncols);
    int i = 0;
    while (i < nrows * ncols) {
        if (i % ncols == (ncols - 1))
            fprintf(fp, "%.f\n", pMat[i++]);
        else
            fprintf(fp, "%.f ", pMat[i++]);
    }
    fclose(fp);
}

int main(int argc, char* argv[]) {
    int p, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request sRequest, rRequest[20];
    double startTime = MPI_Wtime();
    if (p == 1) {
        FILE* fp = fopen(argv[1], "r");;
        int i = 0, j, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float* pMat = (float*)calloc(nrows * ncols, sizeof(float));
        float* pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%f", &pMat[i++]) == 1);
        fclose(fp);
        memcpy(pMatRes, pMat, nrows*ncols*sizeof(float));
        for (int loop = atoi(argv[3]); loop--;) {
            i = 2;
            int endRow = nrows;
            while (i < endRow) {
                int posRes = (i * ncols) - 2;
                int posA = ((i - 1) * ncols) - 3;
                int posB = ((i - 1) * ncols) - 2;
                int posC = ((i - 1) * ncols) - 1;
                int posD = (i * ncols) - 3;
                int posE = (i * ncols) - 1;
                int posF = ((i + 1) * ncols) - 3;
                int posG = ((i + 1) * ncols) - 2;
                int posH = ((i + 1) * ncols) - 1;
                for (j = posRes; j > (posRes - ncols) + 2; j--) {
                    pMatRes[j] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[j]) / 9;
                }
                i++;
            }
            memcpy(pMat, pMatRes, nrows * ncols * sizeof(float));
        }
        writeFile(pMat, argv[2], nrows, ncols);
        double endTime = MPI_Wtime();
        printf("Time: %lf\n", endTime - startTime);
    }

    else if (rank == 0) {
        FILE* fp = fopen(argv[1], "r");
        int i = 0, j, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float* pMat = (float*)calloc(nrows * ncols, sizeof(float));
        float* pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%f", &pMat[i++]) == 1);
        fclose(fp);
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int* spreadRows = (int*)calloc(p, sizeof(int));
        int groupRows = (nrows - 2) / p;
        for (i = p; i--;)
            spreadRows[i] = groupRows;
        int remainRows = (nrows - 2) % p;
        for (i = remainRows; i--;) {
            spreadRows[i]++;
        }
        int rowPos = spreadRows[rank];
        for (i = 1; i < p; i++) {
            MPI_Send(&spreadRows[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Isend(&(pMat[rowPos * ncols]), (spreadRows[i] + 2) * ncols, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &sRequest);
            MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
            rowPos += spreadRows[i];
        }
        memcpy(pMatRes, pMat, nrows * ncols * sizeof(float));
        for (int loop = atoi(argv[3]); loop--;) {
            i = 2;
            while (i < spreadRows[rank] + 2) {
                int posRes = (i * ncols) - 2;
                int posA = ((i - 1) * ncols) - 3;
                int posB = ((i - 1) * ncols) - 2;
                int posC = ((i - 1) * ncols) - 1;
                int posD = (i * ncols) - 3;
                int posE = (i * ncols) - 1;
                int posF = ((i + 1) * ncols) - 3;
                int posG = ((i + 1) * ncols) - 2;
                int posH = ((i + 1) * ncols) - 1;
                for (j = posRes; j > (posRes - ncols) + 2; j--) {
                    pMatRes[j] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[j]) / 9;
                }
                i++;
            }
            memcpy(pMat, pMatRes, nrows * ncols * sizeof(float));
            MPI_Isend(&(pMat[spreadRows[0] * ncols]), ncols, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &sRequest);
            MPI_Irecv(&(pMat[(spreadRows[0] + 1) * ncols]), ncols, MPI_FLOAT, 1, 2, MPI_COMM_WORLD, &rRequest[rank]);
            MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
            MPI_Wait(&rRequest[rank], MPI_STATUS_IGNORE);
        }
        int pos = spreadRows[rank] + 1;
        for (i = 1; i < p; i++) {
            MPI_Irecv(&(pMat[pos * ncols]), spreadRows[i] * ncols, MPI_FLOAT, i, 3, MPI_COMM_WORLD, &rRequest[i]);
            MPI_Wait(&rRequest[i], MPI_STATUS_IGNORE);
            pos += spreadRows[i];
        }
        writeFile(pMat, argv[2], nrows, ncols);
        printf("RANK %d: completed\n", rank);
        double endTime = MPI_Wtime();
        printf("Time: %lf\n", endTime - startTime);
    }
    else {
        int i, j, ncols, groupRows = 0;
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Recv(&groupRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        float* pMatBuf = (float*)calloc((groupRows + 2) * ncols, sizeof(float));
        float* pMatRes = (float*)calloc((groupRows + 2) * ncols, sizeof(float));
        MPI_Irecv(&(pMatBuf[0]), (groupRows + 2) * ncols, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &rRequest[rank]);
        MPI_Wait(&rRequest[rank], MPI_STATUS_IGNORE);
        memcpy(pMatRes, pMatBuf, (groupRows+2)* ncols * sizeof(float));
        for (int loop = atoi(argv[3]); loop--;) {
            i = 2;
            while (i < groupRows + 2) {
                int posRes = (i * ncols) - 2;
                int posA = ((i - 1) * ncols) - 3;
                int posB = ((i - 1) * ncols) - 2;
                int posC = ((i - 1) * ncols) - 1;
                int posD = (i * ncols) - 3;
                int posE = (i * ncols) - 1;
                int posF = ((i + 1) * ncols) - 3;
                int posG = ((i + 1) * ncols) - 2;
                int posH = ((i + 1) * ncols) - 1;
                for (j = posRes; j > (posRes - ncols) + 2; j--) {
                    pMatRes[j] = (pMatBuf[posA--] + pMatBuf[posB--] + pMatBuf[posC--] + pMatBuf[posD--] + pMatBuf[posE--] + pMatBuf[posF--] + pMatBuf[posG--] + pMatBuf[posH--] + pMatBuf[j]) / 9;
                }
                i++;
            }
            memcpy(pMatBuf, pMatRes, (groupRows+2)* ncols * sizeof(float));
            if (rank + 1 == p) {
                MPI_Isend(&(pMatBuf[ncols]), ncols, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &sRequest);
                MPI_Irecv(&(pMatBuf[0]), ncols, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &rRequest[rank - 1]);
                MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&rRequest[rank - 1], MPI_STATUS_IGNORE);
            }

            else {
                MPI_Isend(&(pMatBuf[ncols]), ncols, MPI_FLOAT, rank - 1, 2, MPI_COMM_WORLD, &sRequest);
                MPI_Isend(&(pMatBuf[groupRows * ncols]), ncols, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &sRequest);
                MPI_Irecv(&(pMatBuf[0]), ncols, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, &rRequest[rank - 1]);
                MPI_Irecv(&(pMatBuf[(groupRows + 1) * ncols]), ncols, MPI_FLOAT, rank + 1, 2, MPI_COMM_WORLD, &rRequest[rank + 1]);
                MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
                MPI_Wait(&rRequest[rank - 1], MPI_STATUS_IGNORE);
                MPI_Wait(&rRequest[rank + 1], MPI_STATUS_IGNORE);
            }
        }
        MPI_Isend(&(pMatBuf[ncols]), groupRows * ncols, MPI_FLOAT, 0, 3, MPI_COMM_WORLD, &rRequest[rank]);
        MPI_Wait(&rRequest[rank], MPI_STATUS_IGNORE);
        printf("RANK %d: completed\n", rank);
    }
    MPI_Finalize();
}