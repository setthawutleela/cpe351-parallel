#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#pragma warning (disable : 4996)
#pragma GCC optimize ("O2")

static inline void quickSort(double arr[], int low, int high) {
    if (low < high) {
        register double pivot = arr[high];
        register int i = low - 1, j;
        register double temp;
        for(j = low; j <= high - 1; j++){
            if(arr[j] < pivot){
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        temp = arr[i+1];
        arr[i+1] = arr[high];
        arr[high] = temp;
        int pi = i + 1;
        #pragma omp task default (none) firstprivate(arr, low, pi)
        {
            quickSort(arr, low, pi - 1);
        }
        #pragma omp task default (none) firstprivate(arr, high, pi)
        {
            quickSort(arr, pi + 1, high);
        }
    }
}

static inline void quickSortProc(double arr[], int low, int high, int numprocs, int rank){
    MPI_Request request[4];
    if(numprocs > 1){
        if(low < high){
            register double pivot = arr[high];
            register int i = low - 1, j;
            register double temp;
            for(j = low; j <= high - 1; j++){
                if(arr[j] < pivot){
                    i++;
                    temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
            temp = arr[i+1];
            arr[i+1] = arr[high];
            arr[high] = temp;
            int pi = i + 1;

            // quickSort(arr, low, pi - 1);
            #pragma omp parallel
            {
                #pragma omp single nowait
                quickSort(arr, low, pi - 1);
                #pragma omp taskwait
            }

            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    int n = high - pi;
                    int remainprocs = numprocs - 1;
                    MPI_Isend(&n, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, &request[0]);
                    MPI_Isend(&remainprocs, 1, MPI_INT, rank + 1, 1, MPI_COMM_WORLD, &request[1]);
                    MPI_Isend(&(arr[pi + 1]), n, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &request[2]);
                    MPI_Waitall(3, &(request[0]), MPI_STATUS_IGNORE);
                    MPI_Irecv(&(arr[pi + 1]), n, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD, &request[3]);
                    MPI_Wait(&request[3], MPI_STATUS_IGNORE);
                }
                #pragma omp taskwait
            }
        }
    }
    else{
        #pragma omp parallel
        {
            #pragma omp single nowait
            quickSort(arr, low, high);
            #pragma omp taskwait
        }
    }
}


int main(int argc, char* argv[]) {
    int p, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // double startTime = MPI_Wtime();
    omp_set_num_threads(atoi(argv[3]));

    // # processors = 1
    if (p == 1) {
        // Read file
        FILE* fp = fopen(argv[1], "r");
        register int i, j;
        int n;
        fscanf(fp, "%d", &n);
        double* pNumber = (double*)calloc(n, sizeof(double));
        i = 0;
        while (fscanf(fp, "%lf", &pNumber[i++]) == 1);
        fclose(fp);
        
        // Quick sort

        #pragma omp parallel
        {
            #pragma omp single nowait
            quickSort(pNumber, 0, n - 1);
            #pragma omp taskwait
        }

        // Write file
        fp = fopen(argv[2], "w");
        fprintf(fp, "%d\n", n);
        i = 0;
        while (i < n)
            fprintf(fp, "%.4lf\n", pNumber[i++]);
        fclose(fp);
    }

    else{

        if(rank == 0){
            // Read file
            FILE* fp = fopen(argv[1], "r");
            register int i, j;
            int n;
            fscanf(fp, "%d", &n);
            double* pNumber = (double*)calloc(n, sizeof(double));
            i = 0;
            while (fscanf(fp, "%lf", &pNumber[i++]) == 1);
            fclose(fp);

            quickSortProc(pNumber, 0, n - 1, p, rank);
            
            // Write file
            fp = fopen(argv[2], "w");
            fprintf(fp, "%d\n", n);
            i = 0;
            while (i < n)
                fprintf(fp, "%.4lf\n", pNumber[i++]);
            fclose(fp);
        }

        else{
            MPI_Request request[4];
            int n;
            int remainprocs;
            MPI_Irecv(&n, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(&remainprocs, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &request[1]);
            MPI_Waitall(2, &(request[0]), MPI_STATUS_IGNORE);
            double *pNumber = (double*) calloc(n, sizeof(double));
            MPI_Irecv(pNumber, n, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD, &request[2]);
            MPI_Wait(&request[2], MPI_STATUS_IGNORE);
            
            quickSortProc(pNumber, 0, n - 1, remainprocs, rank);

            MPI_Isend(pNumber, n, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &request[3]);
            MPI_Wait(&request[3], MPI_STATUS_IGNORE);
        }
    }
    MPI_Finalize();
}