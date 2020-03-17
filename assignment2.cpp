#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#pragma warning (disable : 4996)
#pragma GCC optimize ("O2")

int partition(double arr[], int low, int high) {
    double pivot = arr[high];
    register int i = low - 1, j;
    double temp;

    for (j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    temp = arr[i+1];
    arr[i+1] = arr[high];
    arr[high] = temp;
    return i + 1;
}

void quickSort(double arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        #pragma omp parallel sections
        {
            #pragma omp section
            quickSort(arr, low, pi - 1);

            #pragma omp section
            quickSort(arr, pi + 1, high);
        }
    }
}
int main(int argc, char* argv[]) {
    register int p, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    register int pivotVal = p - 1;
    MPI_Request sRequest;
    MPI_Request *rRequest = (MPI_Request*) calloc(rank, sizeof(MPI_Request));
    double startTime = MPI_Wtime();

    // # processors = 1
    if (p == 1) {
        // Read file
        FILE* fp = fopen(argv[1], "r");
        register int i, j;
        int n;
        fscanf(fp, "%d", &n);
        double* pNumber = (double*)calloc(n, sizeof(double));
        double* pResult = (double*)calloc(n, sizeof(double));
        i = 0;
        while (fscanf(fp, "%lf", &pNumber[i++]) == 1);
        fclose(fp);

        // Create thread
        omp_set_num_threads(atoi(argv[3]));
        
        // Quick sort
        quickSort(pNumber, 0, n - 1);
        
        // Write file
        fp = fopen(argv[2], "w");
        fprintf(fp, "%d\n", n);
        i = 0;
        while (i < n)
            fprintf(fp, "%.4lf\n", pNumber[i++]);
        fclose(fp);
        double endTime = MPI_Wtime();
        printf("TIME USED: %lf SEC\n", (double)(endTime - startTime));
        printf("Rank %d: completed\n", rank);

    }

    // # processors > 1
    else {

        // Create thread
        omp_set_num_threads(atoi(argv[3]));
        
        if(rank == 0){
            // Read file
            FILE* fp = fopen(argv[1], "r");
            register int i, j, n;
            fscanf(fp, "%d", &n);
            double* pNumbers = (double*)calloc(n, sizeof(double));
            double* pResult = (double*) calloc(n, sizeof(double));
            i = 0;
            while (fscanf(fp, "%lf", &pNumbers[i++]) == 1);
            fclose(fp);

            // Deligate the tasks to other ranks
            register int * spreadTask = (int*) calloc(p, sizeof(int));
            int tasks = n / p;
            for(i = p; i--;){
                spreadTask[i] = tasks;
            }
            int remainTask = n % p;
            for(i = remainTask; i--;)
                spreadTask[i]++;

            // printf("Rank %d: # Tasks = %d\n", rank, spreadTask[0]);
            double *partNumList = (double*) calloc(spreadTask[0], sizeof(double));
            memcpy(partNumList, pNumbers, spreadTask[0] * sizeof(double));

            // Send # tasks and number list to each rank
            register int pos = spreadTask[0];
            for(i = 1; i < p; i++){
                MPI_Send(&spreadTask[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Isend(&(pNumbers[pos]), spreadTask[i], MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &sRequest);
                MPI_Wait(&sRequest, MPI_STATUS_IGNORE);
                pos += spreadTask[i];
            }
            
            free(pNumbers);

            // printf("Rank %d: List of Number:\n\t", rank);
            // for(i = 0; i < spreadTask[0]; i++)
            //     printf("%.4lf  ", partNumList[i]);
            // printf("\n");

            // Quick Sort
            quickSort(partNumList, 0, spreadTask[0] - 1);

            // printf("Rank %d: List of Sorted Number:\n\t", rank);
            // for(i = 0; i < spreadTask[0]; i++)
            //     printf("%.4lf  ", partNumList[i]);
            // printf("\n");


            // Select samples from its sorted sublist and receive others from other ranks
            double *pSample = (double*) calloc(p * p,  sizeof(double));
            // printf("Rank %d: List of Samples:\n\t", rank);
            for(i = 0; i <= pivotVal; i++){
                pSample[i] = partNumList[(i*spreadTask[0])/(p*p)];
                // printf("%.4lf  ", pSample[i]);
            }
            // printf("\n");

            pos = p;
            for(i = 1; i < p; i++){
                MPI_Irecv(&(pSample[pos]), p, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &rRequest[rank]);
                MPI_Wait(&rRequest[rank], MPI_STATUS_IGNORE);
                pos += p;
            }

            // printf("Rank %d: List of All Samples:\n\t", rank);
            // for(i = 0; i < p*p; i++)
            //     printf("%.4lf  ", pSample[i]);
            // printf("\n");

            // Quick sort the list of all samples
            quickSort(pSample, 0, (p*p) - 1);
            // printf("Rank %d: List of All Sorted Samples:\n\t", rank);
            // for(i = 0; i < p*p; i++)
            //     printf("%.4lf  ", pSample[i]);
            // printf("\n");

            // Select P-1 pivot values
            register double *pivotValueList = (double *) calloc(pivotVal , sizeof(double));
            register int *pivotPos = (int *) calloc(pivotVal , sizeof(int));
            time_t t;
            srand((unsigned) time(&t));
            for(i = 0; i < pivotVal; i++)
                pivotValueList[i] = pSample[rand() % (p*p)];
            quickSort(pivotValueList, 0, pivotVal - 1);
            // printf("Rank %d: List of All Randomized Samples:\n\t", rank);
            // for(i = 0; i < pivotVal; i++)
            //     printf("%.4lf  ", pivotValueList[i]);
            // printf("\n");

            MPI_Bcast(&(pivotValueList[0]), pivotVal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            free(pSample);

            // Find the position of all pivot values
            for(i = 0; i < pivotVal; i++){
                for (j = 0; j < spreadTask[0]; j++){
                    if (pivotValueList[i] <  partNumList[j]){
                        pivotPos[i] = j;
                        // printf("Rank %d: Pivot %d = %d\n", rank, i, pivotPos[i]);
                        break;
                    }
                }
            }

            //Find size of each blocks
            register int *blockSize = (int *) calloc(p , sizeof(int));
            register int start = 0, end = 0;
            for(i = 0; i < p; i++){
                if(i != p - 1)
                    end = pivotPos[i];
                else
                    end = spreadTask[rank];
                blockSize[i] = end - start;
                start = end;
            }

            // for(i = 0; i < p; i++)
            //     printf("Rank %d: blockSize[%d] = %d\n", rank, i, blockSize[i]);
            
            // Sum the size of blocks, which have to run in the processor
            int totalBlockSize;
            for(i = 0; i < p; i++)
                MPI_Reduce(&(blockSize[i]), &totalBlockSize, 1, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            // printf("Rank %d: Total Block Size = %d\n", rank, totalBlockSize);

            // Displacement List
            register int *blockSizeList = (int*) calloc(p, sizeof(int));
            register int *disp = (int*) calloc(p, sizeof(int));
            register int temp;
            pos = 0;
            for(i = 0; i < p; i++){
                MPI_Gather(&(blockSize[i]), 1, MPI_INT, blockSizeList, 1, MPI_INT, i, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            for(i = 0; i < p; i++){
                disp[i] = pos;
                // printf("Rank %d: blockSizeList[%d] = %d\n", rank, i, blockSizeList[i]);
                // printf("Rank %d: disp[%d] = %d\n", rank, i, disp[i]);
                pos += blockSizeList[i];
            }

            // Send the number lists following pivot values list
            double *pTemp = (double*) calloc(totalBlockSize, sizeof(double));
            double *pGatherList = (double*) calloc(totalBlockSize, sizeof(double));
            register int *tempBlockSizeList = (int*) calloc(p, sizeof(int));
            register int *tempDisp = (int*) calloc(p, sizeof(int));
            register int k;

            for(i = 0; i < p; i++){
                if(i == rank){
                    for(k = 0; k < p; k++){
                        if(k != rank){
                            for(j = 0; j < p; j++){
                                MPI_Send(&(blockSizeList[j]), 1, MPI_INT, k, 3, MPI_COMM_WORLD);
                                MPI_Send(&(disp[j]), 1, MPI_INT, k, 4, MPI_COMM_WORLD);
                                // MPI_Barrier(MPI_COMM_WORLD);
                            }
                        }
                    }
                    MPI_Gatherv(&(partNumList[pivotPos[i-1]]), blockSizeList[i], MPI_DOUBLE, pTemp, blockSizeList, disp, MPI_DOUBLE, rank, MPI_COMM_WORLD);
                    memcpy(pGatherList, pTemp, totalBlockSize * sizeof(double));
                }
                else{
                    for(j = 0; j < p; j++){
                        MPI_Recv(&(tempBlockSizeList[j]), 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&(tempDisp[j]), 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    MPI_Gatherv(&(partNumList[pivotPos[i-1]]), tempBlockSizeList[rank], MPI_DOUBLE, pTemp, tempBlockSizeList, tempDisp, MPI_DOUBLE, i, MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            
            // printf("Rank %d: List of Divided Number:\n\t", rank);
            // for(i = 0; i < totalBlockSize; i++)
            //     printf("%.4lf  ", pGatherList[i]);
            // printf("\n");

            quickSort(pGatherList, 0, totalBlockSize - 1);
            // printf("Rank %d: List of Divided Sorted Number:\n\t", rank);
            // for(i = 0; i < totalBlockSize; i++)
            //     printf("%.4lf  ", pGatherList[i]);
            // printf("\n");
            
            // Receive the number list from other ranks
            memcpy(pResult, pGatherList, totalBlockSize * sizeof(double));
            pos = totalBlockSize;
            register int tempSize;
            for(i = 1; i < p; i++){
                MPI_Recv(&tempSize, 1, MPI_INT, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&(pResult[pos]), tempSize, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                pos += tempSize;
                // MPI_Barrier(MPI_COMM_WORLD);
            }

            // printf("Rank %d: Result:\n\t", rank);
            // for(i = 0; i < n; i++)
            //     printf("%.4lf  ", pResult[i]);
            // printf("\n");

            // Write file
            fp = fopen(argv[2], "w");
            fprintf(fp, "%d\n", n);
            i = 0;
            while (i < n)
                fprintf(fp, "%.4lf\n", pResult[i++]);
            fclose(fp);
            double endTime = MPI_Wtime();
            printf("TIME USED: %lf SEC\n", (double)(endTime - startTime));
            printf("Rank %d: completed\n", rank);
        }

        else{
            register int i, j, n; // The amount of number in each rank
            
            // Receive # tasks and number list
            MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // printf("Rank %d: # Tasks = %d\n", rank, n);
            double *partNumList = (double*) calloc(n, sizeof(double));
            MPI_Irecv(&(partNumList[0]), n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &rRequest[rank]);
            MPI_Wait(&rRequest[rank], MPI_STATUS_IGNORE);
            // printf("Rank %d: List of Number:\n\t", rank);
            // for(i = 0; i < n; i++)
            //     printf("%.4lf  ", partNumList[i]);
            // printf("\n");

            //Quick Sort
            quickSort(partNumList, 0, n - 1);
            
            // printf("Rank %d: List of Sorted Number:\n\t", rank);
            // for(i = 0; i < n; i++)
            //     printf("%.4lf  ", partNumList[i]);
            // printf("\n");

            //Select samples from its sorted sublist and send to rank 0
            double *pSample = (double*) calloc(p, sizeof(double));
            // printf("Rank %d: List of Samples:\n\t", rank);
            for(i = 0; i <= pivotVal; i++){
                pSample[i] = partNumList[(i*n)/(p*p)];
                // printf("%.4lf  ", partNumList[(i*n)/(p*p)]);
            }
            // printf("\n");
            MPI_Isend(&(pSample[0]), p, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &sRequest);
            MPI_Wait(&sRequest, MPI_STATUS_IGNORE);

            // Keep the position of pivot values
            register double *pivotValueList = (double *) calloc(pivotVal , sizeof(double));
            register int *pivotPos = (int *) calloc(pivotVal , sizeof(int));
            MPI_Bcast(&(pivotValueList[0]), pivotVal, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Find the position of all pivot values
            for(i = 0; i < pivotVal; i++){
                for (j = 0; j < n; j++){
                    if (pivotValueList[i] < partNumList[j]){
                        pivotPos[i] = j;
                        // printf("Rank %d: Pivot %d = %d\n", rank, i, pivotPos[i]);
                        break;
                    }
                }
            }
            
            //Find size of each blocks
            register int *blockSize = (int *) calloc(p , sizeof(int));
            register int start = 0, end = 0;
            for(i = 0; i < p; i++){
                if(i != p - 1)
                    end = pivotPos[i];
                else
                    end = n;
                blockSize[i] = end - start;
                start = end;
            }

            // for(i = 0; i < p; i++)
            //     printf("Rank %d: blockSize[%d] = %d\n", rank, i, blockSize[i]);

            // Sum the size of blocks, which have to run in the processor
            int totalBlockSize;
            for(i = 0; i < p; i++)
                MPI_Reduce(&(blockSize[i]), &totalBlockSize, 1, MPI_DOUBLE, MPI_SUM, i, MPI_COMM_WORLD);
            // printf("Rank %d: Total Block Size = %d\n", rank, totalBlockSize);

            // Displacement List
            register int *blockSizeList = (int*) calloc(p, sizeof(int));
            register int *disp = (int*) calloc(p, sizeof(int));
            register int temp;
            register int pos = 0;
            for(i = 0; i < p; i++){
                MPI_Gather(&(blockSize[i]), 1, MPI_INT, blockSizeList, 1, MPI_INT, i, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
            }

            for(i = 0; i < p; i++){
                disp[i] = pos;
                // printf("Rank %d: blockSizeList[%d] = %d\n", rank, i, blockSizeList[i]);
                // printf("Rank %d: disp[%d] = %d\n", rank, i, disp[i]);
                pos += blockSizeList[i];
            }

            // Send the number lists following pivot values list
            double *pTemp = (double*) calloc(totalBlockSize, sizeof(double));
            double *pGatherList = (double*) calloc(totalBlockSize, sizeof(double));
            register int *tempBlockSizeList = (int*) calloc(p, sizeof(int));
            register int *tempDisp = (int*) calloc(p, sizeof(int));
            register int k;

            for(i = 0; i < p; i++){
                if(i == rank){
                    for(k = 0; k < p; k++){
                        if(k != rank){
                            for(j = 0; j < p; j++){
                                MPI_Send(&(blockSizeList[j]), 1, MPI_INT, k, 3, MPI_COMM_WORLD);
                                MPI_Send(&(disp[j]), 1, MPI_INT, k, 4, MPI_COMM_WORLD);
                                // MPI_Barrier(MPI_COMM_WORLD);
                            }
                        }
                    }
                    MPI_Gatherv(&(partNumList[pivotPos[i-1]]), blockSizeList[i], MPI_DOUBLE, pTemp, blockSizeList, disp, MPI_DOUBLE, rank, MPI_COMM_WORLD);
                    memcpy(pGatherList, pTemp, totalBlockSize * sizeof(double));
                }
                else{
                    for(j = 0; j < p; j++){
                        MPI_Recv(&(tempBlockSizeList[j]), 1, MPI_INT, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Recv(&(tempDisp[j]), 1, MPI_INT, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    MPI_Gatherv(&(partNumList[pivotPos[i-1]]), tempBlockSizeList[rank], MPI_DOUBLE, pTemp, tempBlockSizeList, tempDisp, MPI_DOUBLE, i, MPI_COMM_WORLD);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            // printf("Rank %d: List of Divided Number:\n\t", rank);
            // for(i = 0; i < totalBlockSize; i++)
            //     printf("%.4lf  ", pGatherList[i]);
            // printf("\n");

            quickSort(pGatherList, 0, totalBlockSize - 1);
            // printf("Rank %d: List of Divided Sorted Number:\n\t", rank);
            // for(i = 0; i < totalBlockSize; i++)
            //     printf("%.4lf  ", pGatherList[i]);
            // printf("\n");

            // Send the number list to rank 0;
            MPI_Send(&totalBlockSize, 1, MPI_INT, 0, 5, MPI_COMM_WORLD);
            MPI_Send(&(pGatherList[0]), totalBlockSize, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
            // MPI_Barrier(MPI_COMM_WORLD);
            printf("Rank %d: completed\n", rank);
        }
    }
    MPI_Finalize();
}