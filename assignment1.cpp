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
    double startTime = MPI_Wtime();
    if (p == 1) {
        FILE* fp = fopen(argv[1], "r");;
        int i = 0, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float * pMat = (float*)calloc(nrows * ncols, sizeof(float));
        float * pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%f", &pMat[i++]) == 1);
        fclose(fp);

        int loop = 0;
        while (loop < atoi(argv[3])) {
            // [1] --> [2]
            i = 1;
            int endRow = nrows;
            while (i < endRow) {
                int pos = (i * ncols) - 1;
                int posRes = ((i + 1) * ncols) - 2;
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMatRes[k] += pMat[pos] + pMat[pos - 1] + pMat[pos - 2];
                    pos--;
                }
                i++;
            }

            // [3] --> [2]
            i = nrows; // startRow
            endRow = 2;
            while (i > endRow) {
                int pos = (i * ncols) - 1;
                int posRes = ((i - 1) * ncols) - 2;
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMatRes[k] += pMat[pos] + pMat[pos - 1] + pMat[pos - 2];
                    pos--;
                }
                i--;
            }

            // [2] --> [2]
            i = 2;
            endRow = nrows;
            while (i < endRow) {
                int posRes = (i * ncols) - 2;
                for (int k = posRes; k > (posRes - ncols) + 2; k--) {
                    pMatRes[k] += pMat[k + 1] + pMat[k - 1];
                    pMatRes[k] = pMatRes[k] / 9;
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
            float * pMatRes = pMat;

        }
        writeFile(pMat, argv[2], nrows, ncols);
        double endTime = MPI_Wtime();
        printf("Time: %lf\n", endTime - startTime);
    }
    else if(rank == 0){
        FILE* fp = fopen(argv[1], "r");
        int i = 0, nrows, ncols;
        fscanf(fp, "%d %d", &nrows, &ncols);
        float * pMat = (float*)calloc(nrows * ncols, sizeof(float));
        while (fscanf(fp, "%lf", &pMat[i++]) == 1);
        fclose(fp);
    }
    else {

    }
    /* Write File*/
    MPI_Finalize();
}