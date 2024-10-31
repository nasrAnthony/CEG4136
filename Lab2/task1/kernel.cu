
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>


__global__ void interleavedPairsSum(int* g_idata, int* g_odata, int n)
{
    int tid = threadIdx.x;

    int idx = blockDim.x * blockIdx.x + tid; 

	int *idata = g_idata + blockIdx.x * blockDim.x;

	// boundary check
	if(idx >= n) return;
	
	// in-place reduction in global memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
		idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = idata[0];
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

	printf("Sum: %d\n", sum);
}

int main()
{
    const int size = 512; // Max number for compatiblity.

	int blocksize = 512;

	dim3 block (blocksize,1);
	dim3 grid ((size+block.x-1)/block.x,1);

	size_t bytes = size * sizeof(int);
	int *h_idata = (int *) malloc(bytes);
	int *h_odata = (int *) malloc(grid.x*sizeof(int));
	int *tmp = (int *) malloc(bytes);

    //int hostArray[size]; //{1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16};

    //generateArray(h_idata, arraySize);

	for (int i = 0; i < size; i++) {
		// mask off high 2 bytes to force max number to 255
		h_idata[i] = (int)(rand() & 0xFF);
	}
	memcpy (tmp, h_idata, bytes);

	// Allocate Device Memory
	int *d_idata = NULL;
	int *d_odata = NULL;
	cudaMalloc((void **) &d_idata, bytes);
	cudaMalloc((void **) &d_odata, grid.x*sizeof(int));

	cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

    //int block = arraySize;
    //int grid = 1;

    interleavedPairsSum<<<grid, block>>>(d_idata, d_odata, size);
    
    cudaDeviceSynchronize();

    // Copy the result back to host
	int sum = 0;
	cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < grid.x; i++) sum += h_odata[i];

    // Output the generated array and the sum
    // printf("Generated Array: ");
    //for (int i = 0; i < arraySize; ++i) {
    //    printf("%d ", hostArray[i]);
   // }
    printf("\nSum of array elements: %d\n", sum);
	checkArray(h_idata, size);

    // Free device memory
    /// free host memory
	free(h_idata);
	free(h_odata);
	// free device memory
	cudaFree(d_idata);
	cudaFree(d_odata);

    return 0;
}

