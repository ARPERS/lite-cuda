#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "AES/AES_encrypt_cpu.cpp"
#include "AES/AES_encrypt_gpu.cu"
#include "AES/AES_decrypt_cpu.cpp"
#include "AES/AES_decrypt_gpu.cu"
#include "AES/AES.cu"

#include "lite_utils.cu"

using namespace std;

///////////////////////////////////////
//1. Encrypt-Decrypt Function
///////////////////////////////////////
__global__ void ltEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N){ // global handle N encryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index*4 + 3 < N)
        AES_encrypt_gpu(ct+index*4, pt+index*4, rek, Nr); // device handle 128 bit encryption
}

__global__ void ltDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N){ // global handle N encryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index*4 + 3 < N)
        AES_decrypt_gpu(pt+index*4, ct+index*4, rek, Nr); // device handle 128 bit encryption
}
void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N){ // run encrypt for all elements
    for(int i=0;i<N;i+=4){
      AES_encrypt_cpu(ct+i, pt+i, rek, Nr);
    }
}
void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N){ // run encrypt for all elements
    for(int i=0;i<N;i+=4){
      AES_decrypt_cpu(pt+i, ct+i, rek, Nr);
    }
}

///////////////////////////////////////
//2. MAIN Lite's Vector Addition
///////////////////////////////////////
__global__ void vectorAddition(uint *d_enc_result, uint *d_enc_a, uint *d_enc_b, int N, uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index * 4 < N){
        uint d_a[4], d_b[4];
        uint *d_result = new uint[4];

        // GPU Decrypt
        AES_decrypt_gpu(d_a, d_enc_a + index*4, d_dec_sched, Nr); 
        AES_decrypt_gpu(d_b, d_enc_b + index*4, d_dec_sched, Nr);  

        if(is_float){
            float *d_f_a = new float[4];
            float *d_f_b = new float[4];
            float *d_f_result = new float[4];
            d_f_a = uintToFloat(d_a);
            d_f_b = uintToFloat(d_b);
            for(int i = 0; i < 4; i ++){
                d_f_result[i] = d_f_a[i] + d_f_b[i];
            }
            d_result = floatToUint(d_f_result);
        }else{
            for(int i = 0; i < 4; i ++){
                d_result[i] = d_a[i] + d_b[i];
            }
        }
        // GPU Encrypt
        AES_encrypt_gpu(d_enc_result + index*4, d_result, d_enc_sched, Nr);
    }
}
// wrapper vector addtion for CPU-GPU comm.
void ltVectorAddition(uint *result, uint *a, uint *b, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_float){
    // Check size, pad so it's divisible by 4
    int padSizeA = padArray(a, N);
    int padSizeB = padArray(b, N);
    
    N += padSizeA; // assuming the size is the same
    
    // CPU Encrypt N elements
    uint *enc_a = new uint[N];
    uint *enc_b = new uint[N];
    uint *enc_result = new uint[N];
    ltEncryptCPU(enc_a, a, enc_sched, Nr, N);
    ltEncryptCPU(enc_b, b, enc_sched, Nr, N);

    // CPU -> GPU: Data
    uint *d_enc_a, *d_enc_b, *d_enc_result;
    size_t size = sizeof(uint)*N;
    gpuErrchk( cudaMalloc(&d_enc_a, size) );
    gpuErrchk( cudaMalloc(&d_enc_b, size) );
    gpuErrchk( cudaMalloc(&d_enc_result, size) );
    gpuErrchk( cudaMemcpy(d_enc_a, enc_a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_enc_b, enc_b, size, cudaMemcpyHostToDevice) );

    // CPU -> GPU: Key
    uint *d_enc_sched;
    uint *d_dec_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );
    
    vectorAddition<<<1, N/4>>>(d_enc_result, d_enc_a, d_enc_b, N, d_enc_sched, d_dec_sched, Nr, is_float);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(enc_result, d_enc_result, size, cudaMemcpyDeviceToHost) );

    // CPU Decrypt
    ltDecryptCPU(result, enc_result, dec_sched, Nr, N);

    removePadArray(a, N, padSizeA);
    removePadArray(b, N, padSizeB);
    N -= padSizeA;
}
// wrapper vector addtion for uint array
void ltVectorAddition(uint *result, uint *a, uint *b, int N, uint *enc_sched, uint *dec_sched, int Nr){
   ltVectorAddition(result, a, b, N, enc_sched, dec_sched, Nr, false);
}
// wrapper vector addtion for float array
void ltVectorAddition(float *result, float *a, float *b, int N, uint *enc_sched, uint *dec_sched, int Nr){

    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);

    ltVectorAddition(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}

///////////////////////////////////////
//3. MAIN LITE's Matrix Multiplication
///////////////////////////////////////
__global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N, uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float){
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

        AES_decrypt_gpu(As[threadIdx.y], As[threadIdx.y], d_dec_sched, Nr); 
        AES_decrypt_gpu(Bs[threadIdx.y], Bs[threadIdx.y], d_dec_sched, Nr); 
        
        __syncthreads();

        // Tile-wise matrix multiplication
        for (int i = 0; i < tile_size; ++i){
            if(is_float){
                tempTotalFloat += *uintToFloat(&As[threadIdx.y][i]) * *uintToFloat(&Bs[i][threadIdx.x]);
                printf("%f %f %f %u\n",*uintToFloat(&As[threadIdx.y][i]), *uintToFloat(&Bs[i][threadIdx.x]), tempTotalFloat, *floatToUint(&tempTotalFloat));
            }else{
                tempTotalUint += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
        }
        // To ensure all data is used before loading the next tiles
        __syncthreads();
    }

    if(is_float){
        printf("~ %u\n",*floatToUint(&tempTotalFloat));
        Cs[threadIdx.y][threadIdx.x] = *floatToUint(&tempTotalFloat);
    }else{
        Cs[threadIdx.y][threadIdx.x] = tempTotalUint;
    }
    
    __syncthreads();

    // debugger
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==1 && blockIdx.y==0){
    //     printf("----Result from GPU before encrypted ----------------\n");
    //     printf("%u %d %d\n", Cvalue, row, col);
    //     for(int i=0;i<4*4;i++) printf("%u ", Cs[i/4][i%4]); printf("\n");
    //     printf("--------------------\n");
    // }
    
    AES_encrypt_gpu(Cs[threadIdx.y], Cs[threadIdx.y], d_enc_sched, Nr); 
    printf("< %u\n",Cs[threadIdx.y][threadIdx.x]);
    // Store the encrypted result in global memory
    C[row * N + col] = Cs[threadIdx.y][threadIdx.x];
}
// wrapper matrix multiplication for CPU-GPU comm.
void ltMatrixMultiplication(uint *result, uint *A, uint *B, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_float){
    const int TILE_SIZE = 4;

    uint *enc_a = new uint[N*N];
    uint *enc_b = new uint[N*N];
    uint *enc_result = new uint[N*N];
    size_t size = sizeof(uint)*N*N;

    printf("----ORI----------------\n");
    for(int i=0;i<N*N;i++) printf("%u ", A[i]); printf("\n");
    for(int i=0;i<N*N;i++) printf("%u ", B[i]); printf("\n");
    printf("-------------------------\n");

    // CPU Encrypt NxN elements
    ltEncryptCPU(enc_a, A, enc_sched, Nr, N*N); 
    ltEncryptCPU(enc_b, B, enc_sched, Nr, N*N);

    printf("----Encrypted----------\n");
    for(int i=0;i<N*N;i++) printf("%u ", enc_a[i]); printf("\n");
    for(int i=0;i<N*N;i++) printf("%u ", enc_b[i]); printf("\n");
    printf("--------------------------\n");

    // CPU -> GPU: Data
    uint *d_enc_a, *d_enc_b, *d_enc_result;
    gpuErrchk( cudaMalloc(&d_enc_a, size) );
    gpuErrchk( cudaMalloc(&d_enc_b, size) );
    gpuErrchk( cudaMalloc(&d_enc_result, size) );

    gpuErrchk( cudaMemcpy(d_enc_a, enc_a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_enc_b, enc_b, size, cudaMemcpyHostToDevice) );

    uint *tmp = new uint[N*N];  // debugger
    gpuErrchk( cudaMemcpy(tmp, d_enc_a, size, cudaMemcpyDeviceToHost) ); 
    printf("----A FROM GPU------------\n");
    for(int i=0;i<N*N;i++) printf("%u ", tmp[i]); printf("\n");
    printf("----------------------------\n");

    // CPU -> GPU: Key
    uint *d_enc_sched;
    uint *d_dec_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );

    // Define grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    matrixMultiplication<<<gridSize, blockSize>>>(d_enc_result, d_enc_a, d_enc_b, N, d_enc_sched, d_dec_sched, Nr, is_float);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(enc_result, d_enc_result, size, cudaMemcpyDeviceToHost) );

    printf("----Result Encrypted from GPU-------\n");
    for(int i=0;i<N*N;i++) printf("%u ", enc_result[i]); printf("\n");
    printf("------------------------------------\n");

    // CPU Decrypt
    ltDecryptCPU(result, enc_result, dec_sched, Nr, N*N);

    printf("----Result Decrypted-------\n");
    for(int i=0;i<N*N;i++) printf("%u ", result[i]); printf("\n");
    printf("------------------------------------\n");
}
// wrapper matrix multiplication for uint matrix
void ltMatrixMultiplication(uint *result, uint *A, uint *B, int N, uint *enc_sched, uint *dec_sched, int Nr){
    ltMatrixMultiplication(result, A, B, N, enc_sched, dec_sched, Nr, false);
}
// wrapper matrix multiplication for float matrix
void ltMatrixMultiplication(float *result, float *A, float *B, int N, uint *enc_sched, uint *dec_sched, int Nr){
    // Float array to uint array
    uint *uint_a = new uint[N*N];
    uint *uint_b = new uint[N*N];
    uint *uint_result = new uint[N*N];

    printf("----Float ORI----------\n");
    for(int i=0;i<N*N;i++) printf("%.2f ", A[i]); printf("\n");
    for(int i=0;i<N*N;i++) printf("%.2f ", B[i]); printf("\n");
    printf("-------------------------\n");

    floatToUintCPU(uint_a, A, N*N);
    floatToUintCPU(uint_b, B, N*N);

    ltMatrixMultiplication(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true);
    
    // uint to float
    printf("----Result Decrypted II-------\n");
    for(int i=0;i<N*N;i++) printf("%u ", uint_result[i]); printf("\n");
    printf("------------------------------------\n");
    
    uintToFloatCPU(result, uint_result, N*N);

    printf("----Result Decrypted II-------\n");
    for(int i=0;i<N*N;i++) printf("%u ", uint_result[i]); printf("\n");
    printf("------------------------------------\n");
}