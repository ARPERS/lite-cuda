#include <iostream>
#include <cuda.h>

#define TILE_SIZE 2

// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Allocate shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0f;

    for (int k = 0; k < N / TILE_SIZE; ++k){
        // Load tiles into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (k * TILE_SIZE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * N + col];

        // Synchronize threads to ensure all data is loaded
        __syncthreads();

        // Perform tile-wise matrix multiplication
        for (int i = 0; i < TILE_SIZE; ++i){
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize threads to ensure all data is used before loading the next tiles
        __syncthreads();
    }

    // Store the result in global memory
    C[row * N + col] = Cvalue;
}

void check(float target, float *a, int N){
    bool flag = false;
    for(int i = 0; i < N * N; ++i){
        if(a[i] != target){
            flag = true;
        }
    }
    if(!flag){
        std::cout << "SUCCESS" << std::endl;
    }else{
        std::cout << "FAIL" << std::endl;
    }
}

int main(){
    int N = 4;  // Matrix size
    int size = N * N * sizeof(float);

    // Allocate host 
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // Allocate device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // initialize
    for (int i = 0; i < N * N; ++i){
        h_A[i] = 2.0;
        h_B[i] = 3.0;
    }

    // host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    check(2*3*N, h_C, N);    

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
