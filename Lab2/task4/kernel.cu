#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

__global__ void neighboredPairsNestedOptimized(int *i_arr_global, int *o_arr_global, int offset, int const dim)
{
    // convert global data pointer to the local pointer of this block
    int *idata = i_arr_global + blockIdx.x * dim;

    // stop condition
    if (offset == 1 && threadIdx.x == 0)
    {
        o_arr_global[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // in place reduction
    idata[threadIdx.x] += idata[threadIdx.x + offset];

    // nested invocation to generate child grids
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        neighboredPairsNestedOptimized<<<gridDim.x, offset / 2>>>(i_arr_global, o_arr_global,
                offset / 2, dim);
    }
}

void generateArray(int* arr, int size) {
    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 11; // 0 to 10 inclusive.
    }

}

void checkArray(int* arr, int size) {
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += arr[i];
	}

	printf("Sum from Host: %d\n", sum);
}

int main()
{
    // Trial and error configuration based on Nvidia MX250 (Pascal Architecture).
    int threads = 256;
    int blocks = 256;
    int size = threads * blocks;

	dim3 block (threads, 1);
    dim3 grid  (blocks, 1);

	// allocate host memory
	int* hostInputArray = (int *) malloc(size * sizeof(int));
	int* hostOutputArray = (int *) malloc(grid.x * sizeof(int));

	generateArray(hostInputArray, size);

	// Allocate device memory
	int* deviceInputArray = NULL;
	int* deviceOutputArray = NULL;

	cudaMalloc((void **) &deviceInputArray, size * sizeof(int));
	cudaMalloc((void **) &deviceOutputArray, grid.x * sizeof(int));

	// Copy host memory to device memory
	cudaMemcpy(deviceInputArray, hostInputArray, size * sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	// Launch Kernel
	neighboredPairsNestedOptimized<<<grid, block.x / 2>>>(deviceInputArray, deviceOutputArray, block.x / 2, block.x);

	cudaDeviceSynchronize();

	// Copy Device memory to Host memory
	cudaMemcpy(hostOutputArray, deviceOutputArray, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	// Get and print sum of array elements.

	int sum = 0;

    for (int i = 0; i < grid.x; i++){ 
        sum += hostOutputArray[i];
    } 

	printf("Sum from Device: %d\n", sum);

	checkArray(hostInputArray, size);

	// Free host memory
	free(hostInputArray);
	free(hostOutputArray);

	// Free device memory
	cudaFree(deviceInputArray);
	cudaFree(deviceOutputArray);

    return 0;
}