#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream> 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

using namespace std;

#define BDIMX 32 // 32
#define BDIMY 32 // 16

void buildMatrix(float* start, int& sizeX, int& sizeY) { //helper function to init the matrices before multiplication. 
    for (int i = 0; i < sizeY; i++) {
        for (int j = 0; j < sizeX; j++) {
            start[i * sizeX + j] = static_cast<float>(rand() % 10);
        }
    }
}

void printMatrix(float* matrix, int& sizeX, int& sizeY) { //helper function to showcase matrix 
    for (int i = 0; i < sizeY; i++) {
        for (int j = 0; j < sizeX; j++) {
            cout << matrix[i * sizeX + j] << " "; //convert 2D index to 1D for printing. 
        }
        cout << endl;
    }
}

/*  
    1. A warp performs a coalesced read of a row from a block of the original matrix stored in global memory
    2. The warp writes the data into shared memory using row-major ordering. No bank conflicts for the write.
    3. After all read/write operations in thread block are synched, you have a 2D shared memory array filled with data from global memory. 
    4. The warp reads a column from the 2D shared memory array. Padding added to avoid bank conflicts.
    5. Warp performs coalesced write of data into a row of the transposed matrix stored in global memory.
*/

__global__ void matrixTranspose(float *out, float *in, int nx, int ny) {

    // static shared memory. 
    __shared__ float tile[BDIMY][BDIMX]; 

    // coordinate in original matrix
    unsigned int ix, iy, ti, to;
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;


    // 1D global memory index for original matrix
    ti = iy * nx + ix; 
    
    if (ix < nx && iy < ny) {
        // load data from global memory to shared memory. Writing in row-major order.
        tile[threadIdx.y][threadIdx.x] = in[ti];
    }

    // thread synchronization
    __syncthreads();

    // thread index 
    unsigned int bidx, irow, icol; 

    bidx = threadIdx.y * blockDim.x + threadIdx.x; 

    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow; 

    // linear global memory index for transposed matrix
    to = iy * ny + ix;

    // transpose with boundary test
    if (iy < nx && ix < ny) {
        // store data to global memory from shared memory. Reading in column-major order.
        out[to] = tile[icol][irow];
    }
}

int main()
{
    int x = 1024;
    int y = 1024;
    
    int matrixSize = x * y;

    int matrixMemSize = matrixSize * sizeof(float);

    // Allocate memory on host
    float* host_matrix = (float*)malloc(matrixMemSize);
    float* host_matrix_transposed = (float*)malloc(matrixMemSize);

    // Build matrix
    buildMatrix(host_matrix, x, y);

    // Print matrix before transpose
    //cout << "Matrix before transpose:" << endl;
    //printMatrix(host_matrix, x, y);
    
    // Allocate device memory
    float* dev_matrix;
    float* dev_matrix_transposed;

    cudaMalloc((void**)&dev_matrix, matrixMemSize);
    cudaMalloc((void**)&dev_matrix_transposed, matrixMemSize);

    // Copy matrices to device
    cudaMemcpy(dev_matrix, host_matrix, matrixMemSize, cudaMemcpyHostToDevice);

    // Setup block size
    dim3 block(BDIMX, BDIMY);

    dim3 grid((x + BDIMX- 1) / BDIMX, (y + BDIMY - 1) / BDIMY);

    matrixTranspose << <grid, block >> > (dev_matrix_transposed, dev_matrix, x, y);
    cudaDeviceSynchronize();

    // Copy result back to device
    cudaMemcpy(host_matrix_transposed, dev_matrix_transposed, matrixMemSize, cudaMemcpyDeviceToHost);

    cout << endl;

    // Print matrix after transpose
    //cout << "Matrix after transpose:" << endl;
    //printMatrix(host_matrix_transposed, y, x);

    // Free up space on device
    cudaFree(dev_matrix);
    cudaFree(dev_matrix_transposed);

    // Free up space on host
    free(host_matrix);
    free(host_matrix_transposed);

    return 0;
}
