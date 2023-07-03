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
    int stride = blockDim.x * gridDim.x;
    const int buffsize = 128;
    
    if(index < N){
        __shared__ uint d_a[buffsize];
        __shared__ uint d_b[buffsize];
        __shared__ uint d_result[buffsize];
        
        __shared__ float d_f_a[buffsize];
        __shared__ float d_f_b[buffsize];
        __shared__ float d_f_result[buffsize];

        for(int idx = index; idx < N; idx += stride){
            // printf("%d %d %d %d %d %d\n", threadIdx.x, blockIdx.x, index, stride, idx, idx*4+3);

            // GPU Decrypt
            if(threadIdx.x%4==0){
                AES_decrypt_gpu(d_a + (idx % buffsize), d_enc_a + idx, d_dec_sched, Nr); 
                AES_decrypt_gpu(d_b + (idx % buffsize), d_enc_b + idx, d_dec_sched, Nr);  
            }

            __syncthreads();

            if(is_float){
                d_f_a[threadIdx.x] = *uintToFloat(&d_a[threadIdx.x]);
                d_f_b[threadIdx.x] = *uintToFloat(&d_b[threadIdx.x]);
                d_f_result[threadIdx.x] = d_f_a[threadIdx.x] + d_f_b[threadIdx.x];
                d_result[threadIdx.x] = *floatToUint(&d_f_result[threadIdx.x]);
            }else{
                d_result[threadIdx.x] = d_a[threadIdx.x] + d_b[threadIdx.x];
            }

            __syncthreads();
            
            // GPU Encrypt
            if(threadIdx.x%4==0){
                AES_encrypt_gpu(d_enc_result + idx, d_result + (idx % buffsize), d_enc_sched, Nr);
            }

            __syncthreads();

        }
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
    
    vectorAddition<<<N/128+1, 128>>>(d_enc_result, d_enc_a, d_enc_b, N, d_enc_sched, d_dec_sched, Nr, is_float);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(enc_result, d_enc_result, size, cudaMemcpyDeviceToHost) );

    // CPU Decrypt
    ltDecryptCPU(result, enc_result, dec_sched, Nr, N);

    removePadArray(a, N, padSizeA);
    removePadArray(b, N, padSizeB);
    N -= padSizeA;
    
    cudaFree(d_enc_a);
    cudaFree(d_enc_b);
    cudaFree(d_enc_sched);
    cudaFree(d_dec_sched);
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
//3. MAIN LITE's MATRIX Multiplication
///////////////////////////////////////
__global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N, uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float, bool is_secure){
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

        if(is_secure){
            AES_decrypt_gpu(As[threadIdx.y], As[threadIdx.y], d_dec_sched, Nr); 
            AES_decrypt_gpu(Bs[threadIdx.y], Bs[threadIdx.y], d_dec_sched, Nr); 
            // To ensure all data is decrypted
            __syncthreads();
        }
    
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
        Cs[threadIdx.y][threadIdx.x] = *floatToUint(&tempTotalFloat);
    }else{
        Cs[threadIdx.y][threadIdx.x] = tempTotalUint;
    }
    
   
    if(is_secure){
        AES_encrypt_gpu(Cs[threadIdx.y], Cs[threadIdx.y], d_enc_sched, Nr); 
         // To ensure all data is encrypted
        __syncthreads();
    }

    // Store the encrypted result in global memory
    C[row * N + col] = Cs[threadIdx.y][threadIdx.x];
}
// wrapper matrix multiplication for CPU-GPU comm.
void ltMatrixMultiplication(uint *result, uint *A, uint *B, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_float, bool is_secure){
    const int TILE_SIZE = 4;

    uint *d_a, *d_b, *d_result;
    uint *d_enc_sched; // for secure key
    uint *d_dec_sched; // for secure key
    size_t size = sizeof(uint)*N*N;
    gpuErrchk( cudaMalloc(&d_a, size) );
    gpuErrchk( cudaMalloc(&d_b, size) );
    gpuErrchk( cudaMalloc(&d_result, size) );

    if(is_secure){
        uint *enc_a = new uint[N*N];
        uint *enc_b = new uint[N*N];

        // CPU Encrypt NxN elements
        ltEncryptCPU(enc_a, A, enc_sched, Nr, N*N); 
        ltEncryptCPU(enc_b, B, enc_sched, Nr, N*N);

        // CPU -> GPU: Data
        gpuErrchk( cudaMemcpy(d_a, enc_a, size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_b, enc_b, size, cudaMemcpyHostToDevice) );

        // uint *tmp = new uint[N*N];  // debugger
        // gpuErrchk( cudaMemcpy(tmp, d_enc_a, size, cudaMemcpyDeviceToHost) ); 
        // printf("----A FROM GPU------------\n");
        // for(int i=0;i<N*N;i++) printf("%u ", tmp[i]); printf("\n");
        // printf("----------------------------\n");

        // CPU -> GPU: Key
        size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
        gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
        gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
        gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );
    }else{
        // CPU -> GPU: Data UNSECURE
        gpuErrchk( cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice) );
    }

    // Define grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    matrixMultiplication<<<gridSize, blockSize>>>(d_result, d_a, d_b, N, d_enc_sched, d_dec_sched, Nr, is_float, is_secure);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost) );

    // printf("----Leak Global Memory of The Result-------\n");
    // float *tmp = new float[N*N];
    // for(int i=0;i<N*N;i++) if(is_float) memcpy(&tmp[i], &result[i], sizeof(uint)),  printf("%.4f ", tmp[i]); else printf("%u ",result[i]); printf("\n");
    // printf("-------------------------------------------\n");

    if(is_secure){
        // CPU Decrypt
        ltDecryptCPU(result, result, dec_sched, Nr, N*N);
    }
}
// wrapper matrix multiplication for uint matrix
void ltMatrixMultiplication(uint *result, uint *A, uint *B, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true){
    ltMatrixMultiplication(result, A, B, N, enc_sched, dec_sched, Nr, false, is_secure);
}
// wrapper matrix multiplication for float matrix
void ltMatrixMultiplication(float *result, float *A, float *B, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true){
    // Float array to uint array
    uint *uint_a = new uint[N*N];
    uint *uint_b = new uint[N*N];
    uint *uint_result = new uint[N*N];

    floatToUintCPU(uint_a, A, N*N);
    floatToUintCPU(uint_b, B, N*N);

    ltMatrixMultiplication(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, is_secure);
    
    // uint to float
    uintToFloatCPU(result, uint_result, N*N);
}