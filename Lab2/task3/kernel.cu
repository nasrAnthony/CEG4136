
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define N 1024 //Total of 1024 elements in the array
#define Nth 256 //Total number of threads per block

using namespace std;



__global__ void sumWithNestedExecutionKernel(int *input, int *output, int n)
{
    int tid = threadIdx.x; //local id within the block
    int gtid = blockIdx.x * blockDim.x * 2 + tid;
    extern __shared__ int sharedarr[]; //to be used by child kernel calls
    //populate the array load 2 elem / thread
    sharedarr[tid] = (gtid < n ? input[gtid] : 0) + (gtid + blockDim.x < n ? input[gtid + blockDim.x] : 0);
    __syncthreads();

    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (tid < step) {
            sharedarr[tid] += sharedarr[tid + step];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedarr[0];
    }

    if (tid == 0 && blockIdx.x == 0) {
        int nBlocks = (n + (blockDim.x * 2 - 1)) / (blockDim.x * 2);
        if (nBlocks > 1) { //nested call is here... 
            sumWithNestedExecutionKernel << <nBlocks, blockDim.x, blockDim.x * sizeof(int) >> > (output, output, nBlocks);
        }
    }


}

int main()
{
    int arr_in[N];
    int arr_out;
    int* dev_arr;
    int* dev_arr_out;
    
    //fill the array with random integers... 
    for (int i = 0; i < N; i++) {
        arr_in[i] = rand() % 100; //values between 0 -> 100
    }


    //allocate memory on device 
    cudaMalloc((void**)&dev_arr, N * sizeof(int));
    cudaMalloc((void**)&dev_arr_out, N * sizeof(int));
    //copy array values from host memory to device memory
    cudaMemcpy(dev_arr, arr_in, sizeof(int) * N, cudaMemcpyHostToDevice);
    int numberBlocks = ((Nth * 2 - 1) + N) / (Nth * 2); //get number of blocks

    //kernel call
    sumWithNestedExecutionKernel << <numberBlocks, Nth, Nth * sizeof(int) >> > (dev_arr, dev_arr_out, N);
    cudaMemcpy(&arr_out, dev_arr_out, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Total sum of the array is: " << arr_out << std::endl;

    cudaFree(dev_arr);
    cudaFree(dev_arr_out);

    return 0;
}
