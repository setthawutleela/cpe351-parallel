#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>

int main(int argc, char **argv){
    float *a_h, *b_h;   // Host data
    float *a_d, *b_d;   // Device data
    int N = 14, nBytes, i;

    printf("Start allocating\n");
    nBytes = N * sizeof(float);

    printf("Allocating in Host\n");
    a_h = (float*) malloc(nBytes);
    b_h = (float*) malloc(nBytes);

    printf("Allocating in Device\n");
    cudaMalloc((void**)&a_d, nBytes);
    cudaMalloc((void**)&b_d, nBytes);

    printf("End allocating\n");

    for(i=0; i<N; i++)
        a_h[i] = 100.0 + i;

    printf("Start memcpy\n");
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
    
    printf("End memcpy\n");

    for(i=0; i<N; i++)
        assert(a_h[i] == b_h[i]);
    
    free(a_h);
    free(b_h);
    cudaFree(a_d);
    cudaFree(b_d);
	return 0;
}