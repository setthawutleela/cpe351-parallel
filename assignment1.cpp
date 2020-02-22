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
        int i = 0, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float* pMat = (float*)calloc(nrows * ncols, sizeof(float));
        float* pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%f", &pMat[i++]) == 1);
        fclose(fp);

        int loop = 0;

        while (loop < atoi(argv[3])) {
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
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMatRes[k] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[k]) / 9;
                }
                i++;
            }

            i = 2;
            endRow = nrows;
            while (i < endRow) {
                int posRes = (i * ncols) - 2;
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMat[k] = pMatRes[k];
                }
                i++;
            }
            loop++;
            float* pMatRes = pMat;
        }
        writeFile(pMat, argv[2], nrows, ncols);
        double endTime = MPI_Wtime();
        printf("Time: %lf\n", endTime - startTime);
    }
    else if (p > 1) {
        if (rank == 0) {
            FILE* fp = fopen(argv[1], "r");;
            int i = 0, j, nrows, ncols;
            fscanf(fp, "%d %d", &nrows, &ncols);
            float* pMat = (float*)calloc(nrows * ncols, sizeof(float));
            float* pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
            while (fscanf(fp, "%f", &pMat[i++]) == 1);
            fclose(fp);
            MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
            int groupRows = (nrows - 2) / p;
            int remainRows = (nrows - 2) - ((nrows - 2) / p) * p;
            printf("groupRows = %d\n", groupRows);
            printf("remainRows = %d\n", remainRows);
            MPI_Bcast(&groupRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    /*else if (rank == 0) {
        FILE* fp = fopen(argv[1], "r");;
        int i = 0, j, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float* pMat = (float*)calloc(nrows * ncols, sizeof(float));
        float* pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%f", &pMat[i++]) == 1);
        fclose(fp);
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int groupRows = (nrows-2)/p;
        int remainRows = (nrows - 2) - ((nrows - 2) / p) * p;
        printf("groupRows = %d\n", groupRows);
        printf("remainRows = %d\n", remainRows);
        MPI_Bcast(&groupRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //printf("RANK %d\n", rank);
        int loop = 0;
        while (loop < atoi(argv[3])) {
            for (i = 1; i < p; i++) {
                int rowPos = i * groupRows * ncols;
                MPI_Send(&(pMat[rowPos]), (groupRows + 2) * ncols, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
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
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMatRes[k] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[k]) / 9;
                }
                i++;
            }
            i = 2;
            while (i < groupRows + 2) {
                int posRes = (i * ncols) - 2;
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMat[k] = pMatRes[k];
                }
                i++;
            }

            if (remainRows >= 1) {
                i = (groupRows * p) + 1;
                while (i < nrows) {
                    int posRes = (i * ncols) - 2;
                    int posA = ((i - 1) * ncols) - 3;
                    int posB = ((i - 1) * ncols) - 2;
                    int posC = ((i - 1) * ncols) - 1;
                    int posD = (i * ncols) - 3;
                    int posE = (i * ncols) - 1;
                    int posF = ((i + 1) * ncols) - 3;
                    int posG = ((i + 1) * ncols) - 2;
                    int posH = ((i + 1) * ncols) - 1;
                    for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                        pMatRes[k] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[k]) / 9;
                    }
                    i++;
                }
                i = (groupRows * p) + 1;
                while (i < nrows) {
                    int posRes = (i * ncols) - 2;
                    for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                        pMat[k] = pMatRes[k];
                    }
                    i++;
                }
            }

            for (i = 1; i < p; i++) {
                MPI_Irecv(&(pMat[((i * groupRows) + 1) * ncols]), groupRows * ncols, MPI_FLOAT, i, 1, MPI_COMM_WORLD, &rRequest[i]);
                MPI_Wait(&rRequest[i], MPI_STATUS_IGNORE);
            }
            loop++;
        }
        writeFile(pMat, argv[2], nrows, ncols);
        double endTime = MPI_Wtime();
        printf("Time: %lf\n", endTime - startTime);
    }
    else {
        int i, j, ncols, groupRows;
        float* pMat = NULL, * pMatBuf = NULL;
        MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&groupRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //printf("RANK %d\n", rank);
        pMatBuf = (float*)calloc((groupRows + 2) * ncols, sizeof(float));
        float* pMatRes = (float*)calloc((groupRows + 2) * ncols, sizeof(float));
        MPI_Recv(&(pMatBuf[0]), (groupRows+2) * ncols, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
            for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                pMatRes[k] = (pMatBuf[posA--] + pMatBuf[posB--] + pMatBuf[posC--] + pMatBuf[posD--] + pMatBuf[posE--] + pMatBuf[posF--] + pMatBuf[posG--] + pMatBuf[posH--] + pMatBuf[k]) / 9;
                //printf("%.f\n", pMatRes[k]);
            }
            i++;
        }

        i = 2;
        while (i < groupRows+2) {
            int posRes = (i * ncols) - 2;
            for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                pMatBuf[k] = pMatRes[k];
            }
            i++;
        }
        MPI_Isend(&(pMatBuf[ncols]), groupRows*ncols, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &sRequest);
    }*/
    MPI_Finalize();
}