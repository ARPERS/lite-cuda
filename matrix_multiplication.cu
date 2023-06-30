#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda.h>

#include "lite.cu"

using namespace std;

__global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N, bool is_float){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_size = 4;

    // Allocate shared memory for tiles
    __shared__ uint As[tile_size][tile_size];
    __shared__ uint Bs[tile_size][tile_size];
    __shared__ uint Cs[tile_size][tile_size];

    uint tempTotalUint = 0;
    float tempTotalFloat = 0.0f;

    for (int k = 0; k < N / tile_size; ++k){
        // Load tiles into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (k * tile_size + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k * tile_size + threadIdx.y) * N + col];
    
        // To ensure all data is loaded
        __syncthreads();

        // Tile-wise matrix multiplication
        for (int i = 0; i < tile_size; ++i){
            if(is_float){
                tempTotalFloat += *uintToFloat(&As[threadIdx.y][i]) * *uintToFloat(&Bs[i][threadIdx.x]);
            }else{
                tempTotalUint += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
        }
        // To ensure all data is used before loading the next tiles
        __syncthreads();
    }

    if(is_float){
        Cs[threadIdx.y][threadIdx.x] = *floatToUint(&tempTotalFloat);;
    }else{
        Cs[threadIdx.y][threadIdx.x] = tempTotalUint;
    }
    
    __syncthreads();

    // Store the encrypted result in global memory
    C[row * N + col] = Cs[threadIdx.y][threadIdx.x];
}

void check(uint *a, uint *b, uint *res, int N){
    bool flag = false;
    uint *c = new uint[N*N];
    for(int i = 0; i < N; ++i) for(int j = 0; j < N; ++j) c[i*N+j] = 0;

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            for(int k = 0; k < N; ++k){
                int row_a = i;
                int col_b = j;
                int cr_ab = k;
                c[row_a*N + col_b] += a[row_a*N + cr_ab]*b[cr_ab*N + col_b];
            }
    

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j){
            if(c[i*N+j]!=res[i*N+j]){
                flag = true;
            }else{
                printf("%d %d %u %d\n", i, j , c[i*N+j], res[i*N+j]);
            }
        }
    if(!flag){
        cout << "SUCCESS" << endl;
    }else{
        cout << "FAIL" << endl;
    }
    printf("ANS: ");
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            printf("%u ", c[i*N+j]);
    printf("\nRES: ");
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            printf("%u ", res[i*N+j]);
    printf("\n");
}

int main(){
    int N = 8;  // Matrix size
    int size = N * N * sizeof(uint);

    // Allocate host 
    uint *h_A = new uint[N * N];
    uint *h_B = new uint[N * N];
    uint *h_C = new uint[N * N];

    // initialize
    for (int i = 0; i < N * N; ++i){
        h_A[i] = rand()%5+1;
        h_B[i] = rand()%10;
    }

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    // Define grid and block dimensions
    dim3 blockSize(4, 4);
    dim3 gridSize(N / 4, N / 4);

    matrixMultiplication<<<gridSize, blockSize>>>(h_C, h_A, h_B, N, false);

    cudaDeviceSynchronize();

    check(h_A, h_B, h_C, N);    
}
