#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// Reduce warp divergence by using interleaved pairs instead of neighboring pairs. 
// Make interleaved pairs a template kernel.
template <typename T>
__global__ void interleavedPairsSum(T* i_arr_global, T* o_arr_global, int size)
{
    int tid = threadIdx.x;

	// Note the "* 2" for loop unrolling
    int idx = blockDim.x * blockIdx.x * 2 + tid; 

	T* i_data = blockIdx.x * blockDim.x * 2 + i_arr_global;

	// unrolling 2 data blocks
	if(idx + blockDim.x < size) {
		i_arr_global[idx] += i_arr_global[idx + blockDim.x];
	}
	
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			i_data[tid] += i_data[tid + stride];
		}
		__syncthreads();
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

		if constexpr (std::is_integral<T>::value) {
			arr[i] = rand() % 11; // 0 to 10 inclusive.
		} 
		
		else if constexpr (std::is_floating_point<T>::value) {
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
    const int size = 512; // Max number for compatiblity.

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

    interleavedPairsSum<<<grid, block>>>(deviceInputArray, deviceOutputArray, size);
    
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

