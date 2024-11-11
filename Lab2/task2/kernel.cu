#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// Reduce warp divergence by using interleaved pairs instead of neighboring pairs. 
// Make interleaved pairs a template kernel.
template <typename T, int blockSize>
__global__ void interleavedPairsSum(T* i_arr_global, T* o_arr_global, int size)
{
    int tid = threadIdx.x;

    int idx = blockDim.x * blockIdx.x + tid; 

	volatile T* i_data = blockIdx.x * blockDim.x + i_arr_global;

	if(idx >= size) {
		return;
	}
	
	// Completely unroll reduction loop.
	if (blockSize >= 512) {
		if (tid < 256) {
			i_data[tid] += i_data[tid + 256];
		}
		__syncthreads();
	}

	if (blockSize >= 256) {
		if (tid < 128) {
			i_data[tid] += i_data[tid + 128];
		}
		__syncthreads();
	}

	if (blockSize >= 128) {
		if (tid < 64) {
			i_data[tid] += i_data[tid + 64];
		}
		__syncthreads();
	}

	// Unroll the last 6 iterations which is the last warp. This means we don't need __syncthreads(). 
	if (tid < 32) {
		if (blockSize >= 64) i_data[tid] += i_data[tid + 32];
		if (blockSize >= 32) i_data[tid] += i_data[tid + 16];
		if (blockSize >= 16) i_data[tid] += i_data[tid + 8];
		if (blockSize >= 8) i_data[tid] += i_data[tid + 4];
		if (blockSize >= 4) i_data[tid] += i_data[tid + 2];
		if (blockSize >= 2) i_data[tid] += i_data[tid + 1];	
	}


	// write result for this block to global mem
	if (tid == 0) {
		o_arr_global[blockIdx.x] = i_data[0];
	}
}

// Generate a generic array.
template <typename T>
void generateArray(T* arr, int size) {
    srand(time(NULL));

    for (int i = 0; i < size; i++) {

		if (std::is_integral<T>::value) {
			arr[i] = rand() % 11; // 0 to 10 inclusive.
		} 
		
		else if (std::is_floating_point<T>::value) {
			arr[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * static_cast<T>(10); // 0.0 to 10.0 inclusive
 		}

    }

}

template <typename T>
void checkArray(T* arr, int size) {
	T sum = 0;
	for (int i = 0; i < size; i++) {
		sum += arr[i];
	}

	printf("Sum from Host: %d\n", sum);
}

int main()
{
	// For block sizes (and therefore array sizes) we are assuming powers of 2. 
	// A common maximum of threads per block is 512, so we are assuming that 512 is the maximum array size. 
	// Therefore, the applicable sizes are powers of 2 up to 512 inclusive. {1, 2, 4, 8, etc.}
    const int size = 512; 

	int grid = 1;
	int block = size;

	size_t bytes = size * sizeof(int);
	int *hostInputArray = (int *) malloc(bytes);
	int *hostOutputArray = (int *) malloc(sizeof(int));

    generateArray(hostInputArray, size);

	// Allocate Device Memory
	int *deviceInputArray = NULL;
	int *deviceOutputArray = NULL;
	cudaMalloc((void **) &deviceInputArray, bytes);
	cudaMalloc((void **) &deviceOutputArray, sizeof(int));

	cudaMemcpy(deviceInputArray, hostInputArray, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	switch (block) {
		case 512:
    		interleavedPairsSum<int, 512><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 256:
    		interleavedPairsSum<int, 256><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 128: 
    		interleavedPairsSum<int, 128><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 64:
    		interleavedPairsSum<int, 64><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 32: 
    		interleavedPairsSum<int, 32><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 16:
    		interleavedPairsSum<int, 16><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 8:
    		interleavedPairsSum<int, 8><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 4:
    		interleavedPairsSum<int, 4><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 2:
    		interleavedPairsSum<int, 2><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
		case 1:
    		interleavedPairsSum<int, 1><<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
			break;
	}


    
    cudaDeviceSynchronize();

    // Copy the result back to host
	cudaMemcpy(hostOutputArray, deviceOutputArray, sizeof(int), cudaMemcpyDeviceToHost);

	int sum = hostOutputArray[0];

    printf("\nSum from Device: %d\n", sum);
	checkArray(hostInputArray, size);

    // Free device memory
    /// free host memory
	free(hostInputArray);
	free(hostOutputArray);
	// free device memory
	cudaFree(deviceInputArray);
	cudaFree(deviceOutputArray);

    return 0;
}

