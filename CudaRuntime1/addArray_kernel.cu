#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>



__global__ void kernel_add_arrays(const int* a, const int* b, int* c) {
	int index = threadIdx.x;
	c[index] = a[index] + b[index];
}

int main() {
	const int arraySize = 5;
	int a[arraySize] = { 1, 2, 3, 4, 5 };
	int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };


	int* dev_a, * dev_b, * dev_c; // init pointer to the device for each array

	//allocate memory space on device
	cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
	cudaMalloc((void**)&dev_c, arraySize * sizeof(int));

	//copy contents of arrays into allocated memory on device
	cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);


	//call kernel
	kernel_add_arrays << < 1, 5 >> > (dev_a, dev_b, dev_c);
	cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < arraySize; i++) {
		printf("%d\n", c[i]);
	}

	return 0;
}

