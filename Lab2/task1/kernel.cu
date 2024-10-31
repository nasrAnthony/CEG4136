
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

__global__ void neighboredPairsSum(int* i_arr_global, int* o_arr_global, int size) {

	// set thread ID
	int tid = threadIdx.x;

	//convert global data pointer to the local pointer of this block
	int* i_arr = blockDim.x * blockIdx.x + i_arr_global;

	// boundary check 
	if (tid >= size) {
		return;
	}

	// in-place reduction in global memory
	for (int offset = 1; offset < blockDim.x; offset *= 2) {

		if ((tid % (2 * offset)) == 0) { 
			i_arr[tid] += i_arr[tid + offset];
		}

		__syncthreads();
	}

	// write result for this block into global memory
	if (tid == 0) {
		o_arr_global[blockIdx.x] = i_arr[0];
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
	int size = 512; // best compatibility for every gpu as we are configuring 1 block with 512 threads where each thread represents an element of the array.

	int grid = 1;
	int block = size;

	// allocate host memory
	int* hostInputArray = (int *) malloc(size * sizeof(int));
	int* hostOutputArray = (int *) malloc(sizeof(int));

	generateArray(hostInputArray, size);

	// Allocate device memory
	int* deviceInputArray = NULL;
	int* deviceOutputArray = NULL;

	cudaMalloc((void **) &deviceInputArray, size * sizeof(int));
	cudaMalloc((void **) &deviceOutputArray, sizeof(int));

	// Copy host memory to device memory
	cudaMemcpy(deviceInputArray, hostInputArray, size * sizeof(int), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	// Launch Kernel
	neighboredPairsSum<<<grid, block>>>(deviceInputArray, deviceOutputArray, size);

	cudaDeviceSynchronize();

	// Copy Device memory to Host memory
	cudaMemcpy(hostOutputArray, deviceOutputArray, sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	// Get and print sum of array elements.
	int sum = hostOutputArray[0];

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

