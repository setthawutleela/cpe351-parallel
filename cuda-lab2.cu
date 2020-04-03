#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 1
#define THREADSIZE_X 16
#define THREADSIZE_Y 16

__global__ void leibniz(float* a_d){

    // Calculate the position in 1D
    int threadRank = threadIdx.y*blockDim.x + threadIdx.x;
    float sign;
    // printf("BLOCK[%d, %d] THREAD[%d, %d] >>> RANK = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadRank);
    
    // Odd = plus sign, Even = minus sign
    if (threadRank%2 == 0)
        sign = 1;
    else
        sign = -1;

    // Find denominator, which is an odd number
    int denominator = (2 * threadRank) + 1;
    // printf("Deniminator = %d\n", denominator);
    a_d[threadRank] = sign/denominator * 4;
}

int main(int argc, char **argv){
    float *a_h;
    float *a_d;
    float pi = 0;

    int size = THREADSIZE_X * THREADSIZE_Y * sizeof(float);
    a_h = (float*) malloc(size);
    cudaMalloc((void**)&a_d, size);

    dim3 BLOCK(BLOCKSIZE, BLOCKSIZE);
    dim3 THREAD(THREADSIZE_X, THREADSIZE_Y);
    leibniz<<<BLOCK, THREAD>>>(a_d);
    cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);

    for(register int i = 0; i < THREADSIZE_X * THREADSIZE_Y; i++)
        pi += a_h[i];

    printf("\nLeibniz formula for pi:\n\t1/1 - 1/3 + 1/5 - 1/7 + 1/9 - ... = pi/4\n\n>>> Pi = %.10f\n", pi);

	return 0;
}