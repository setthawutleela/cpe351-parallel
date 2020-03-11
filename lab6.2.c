#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[])
{
    int nt, rank, i, nrows, ncols;
    clock_t start, end;
    omp_set_num_threads(atoi(argv[4]));
    start = clock();
    // Read matrix A
    FILE* fp = fopen(argv[1], "r");
    i = 0;
    fscanf(fp, "%d %d", &nrows, &ncols);
    float *pMatA=(float*)calloc(nrows*ncols, sizeof(float));
    while(fscanf(fp, "%f", &pMatA[i++]) == 1);
    fclose(fp);
    
    // Read matrix B
    fp = fopen(argv[2], "r");
    i = 0;
    fscanf(fp, "%d %d", &nrows, &ncols);
    float *pMatB=(float*)calloc(nrows*ncols, sizeof(float));
    while(fscanf(fp, "%f", &pMatB[i++]) == 1);
    fclose(fp);
    float *pMatRes=(float*)calloc(nrows*ncols, sizeof(float));
    
    // Calculate
    #pragma omp parallel for
    for(i = 0; i < nrows * ncols; i++)
    {
        pMatRes[i] = pMatA[i] + pMatB[i];
    }
    
    // Write file
    fp = fopen(argv[3], "w");
    fprintf(fp, "%d %d\n", nrows, ncols);
    i = 0;
    while (i < nrows * ncols) {
        if (i % ncols == (ncols - 1))
            fprintf(fp, "%.f\n", pMatRes[i++]);
        else
            fprintf(fp, "%.f ", pMatRes[i++]);
    }
    fclose(fp);
    end = clock();
    #pragma omp single
    {
        printf("TIME USED: %.4f SEC\n", (float)(end-start)/CLOCKS_PER_SEC);
    }
}