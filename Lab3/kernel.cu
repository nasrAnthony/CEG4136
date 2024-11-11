#include <iostream> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

//#define T 2 //32 //Tile size 32
#define T 32

void buildMatrix(int* start, int& size) { //helper function to init the matrices before multiplication. 
	for (int i = 0; i < size; i++) {
		*(start + i) = rand() % 10;
	};
}

void printMatrix(int* start, int& size) { //helper function to showcase matrix 
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			std::cout << *(start + i * size + j) << " "; //convert 2D index to 1D for printing. 
		}
		std::cout << std::endl;
	}
}

//Kernel implementation
__global__ void matrixMultKernel(int* A, int* B, int* C, int size) {
	//shared memory setup
	__shared__ int Ashm[T][T];
	__shared__ int Bshm[T][T];
	//calculate index
	int row = blockIdx.y * T + threadIdx.y;
	int col = blockIdx.x * T + threadIdx.x;

	int val = 0;

	for (int i = 0; i < size / T; i++) {
		Ashm[threadIdx.y][threadIdx.x] = A[row * size + i * T + threadIdx.x];
		Bshm[threadIdx.y][threadIdx.x] = B[(i * T + threadIdx.y) * size + col];
		__syncthreads();
		for (int j = 0; j < T; j++) {
			val += Ashm[threadIdx.y][j] * Bshm[j][threadIdx.x];
		}
		__syncthreads();
	}
	C[row * size + col] = val;
}

int main() {

	//int N = 8;//= 1024;
	int N = 1024;
	int totalSize = N * N;
	//fetch total size of array in bytes
	int size = N * N * sizeof(int);
	//allocate memory on host. 
	int* hostA = (int*)malloc(size);//cast pointer returned from malloc to type int* || operand A
	int* hostB = (int*)malloc(size);//cast pointer returned from malloc to type int* || operand B
	int* hostC = (int*)malloc(size);//cast pointer returned from malloc to type int* || result  C

	//build the matrices with helper function
	buildMatrix(hostA, totalSize); //init matrix A
	buildMatrix(hostB, totalSize); //init matrix B

	//Print matrices A and B
	//std::cout << "Matrix A:" << std::endl;
	//printMatrix(hostA, N);
	//std::cout << "\nMatrix B:" << std::endl;
	//printMatrix(hostB, N);

	//allocate device memory
	int* devA; int* devB; int* devC;
	cudaMalloc((void**)&devA, size);
	cudaMalloc((void**)&devB, size);
	cudaMalloc((void**)&devC, size);

	//copy the matrices to device
	cudaMemcpy(devA, hostA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devB, hostB, size, cudaMemcpyHostToDevice);

	//setup block size
	dim3 blockDim(T, T); //32x32
	dim3 gridDim((N + T - 1) / T, (N + T - 1) / T);


	matrixMultKernel << <gridDim, blockDim >> > (devA, devB, devC, N); //call kernel
	cudaDeviceSynchronize();

	//copy result back to device
	cudaMemcpy(hostC, devC, size, cudaMemcpyDeviceToHost);

	//Print matrix C
	//std::cout << "\nResult Matrix C (A * B):" << std::endl;
	//printMatrix(hostC, N);

	//free up space on device
	cudaFree(devA); cudaFree(devB); cudaFree(devC);
	//free up space on host
	free(hostA), free(hostB); free(hostC);
	return 0;

}
