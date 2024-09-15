#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void myFirstKernel() {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int globalThreadId = blockId * blockDim.x + threadId;
    printf("Block %d, Thread %d, Global Thread %d\n", blockId, threadId, globalThreadId);
}

int main() {
    myFirstKernel << <4, 8 >> > ();
    cudaDeviceSynchronize();
    return 0;
}

