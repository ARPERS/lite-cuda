#include <iostream>
#include <cuda.h>

#include "lite.cu"

#define TILE_SIZE 2

using namespace std;

void check(float target, float *a, int N){
    bool flag = false;
    for(int i = 0; i < N * N; ++i){
        if(a[i] != target){
            flag = true;
        }
    }
    if(!flag){
        cout << "SUCCESS" << endl;
    }else{
        cout << "FAIL" << endl;
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
}
