#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "AES/AES_encrypt_cpu.cpp"
#include "AES/AES_encrypt_gpu.cu"
#include "AES/AES_decrypt_cpu.cpp"
#include "AES/AES_decrypt_gpu.cu"
#include "AES/AES.cu"

using namespace std;

//0. Debugging
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//0. Utils
__device__ unsigned int* floatToUint(float  *input){
    unsigned char  * temp1 = reinterpret_cast<unsigned char  *>(input);       
    unsigned int  * output = reinterpret_cast<unsigned int  *>(temp1);    
    return output;
}
__device__ float * uintToFloat(unsigned int  *input){
    unsigned char  * temp1 = reinterpret_cast<unsigned char  *>(input);       
    float  * output = reinterpret_cast<float  *>(temp1);    
    return output;
}
void floatToUintCPU(uint *dest, const float *source, int N) {
    for(int i=0; i<N; i++) memcpy(&dest[i], &source[i], sizeof(float));
}
void uintToFloatCPU(float *dest, const uint *source, int N) {
    for(int i=0; i<N; i++) memcpy(&dest[i], &source[i], sizeof(uint));
}

//1. Encrypt-Decrypt Function
__device__ void ltEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr){
    AES_encrypt_gpu(ct, pt, rek, Nr);
}

__device__ void ltDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr){
    AES_decrypt_gpu(pt, ct, rek, Nr);
}

void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr){ // encrypt 4 elements pointer+0, +1, +2, +3
      AES_encrypt_cpu(ct, pt, rek, Nr);
}
void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr){ // encrypt 4 elements pointer+0, +1, +2, +3
      AES_decrypt_cpu(pt, ct, rek, Nr);
}
void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N){ // run encrypt for all elements
    for(int i=0;i<N;i+=4){
      ltEncryptCPU(ct+i, pt+i, rek, Nr);
    }
}
void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N){ // run encrypt for all elements
    for(int i=0;i<N;i+=4){
      ltDecryptCPU(pt+i, ct+i, rek, Nr);
    }
}

//2. MAIN Lite's Vector Addition
__global__ void vectorAddition(uint *d_enc_result, uint *d_enc_a, uint *d_enc_b, int N, uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index * 4 < N){
        uint d_a[4], d_b[4];
        uint *d_result = new uint[4];

        // GPU Decrypt
        ltDecryptGPU(d_a, d_enc_a + index*4, d_dec_sched, Nr); 
        ltDecryptGPU(d_b, d_enc_b + index*4, d_dec_sched, Nr);  

        if(is_float){
            float *d_f_a = new float[4];
            float *d_f_b = new float[4];
            float *d_f_result =new float[4];
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
        ltEncryptGPU(d_enc_result + index*4, d_result, d_enc_sched, Nr);
    }
}

// wrapper vector addtion CPU-GPU comm.
void ltVectorAddition(uint *result, uint *a, uint *b, int N, uint *enc_sched, uint *dec_sched, int Nr, bool is_float){
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
}

// wrapper vector addtion for uint array
void ltVectorAddition(uint *result, uint *a, uint *b, int N, uint *enc_sched, uint *dec_sched, int Nr){
   ltVectorAddition(result, a, b, N, enc_sched, dec_sched, Nr, false);
}

// wrapper vector addtion for float array
void ltVectorAddition(float *result, float *a, float *b, int N, uint *enc_sched, uint *dec_sched, int Nr){

    // debug
    // printf("BEFORE\n");
    // for(int i = 0; i < N; i++) printf("%f ", a[i]); printf("\n");

    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);
    floatToUintCPU(uint_result, result, N);

    // debug
    // printf("PUNNED to UINT\n");
    // for(int i = 0; i < N; i++) printf("%u ", uint_a[i]); printf("\n");

    ltVectorAddition(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true);

    // uint to float
    uintToFloatCPU(a, uint_a, N);
    uintToFloatCPU(b, uint_b, N);
    uintToFloatCPU(result, uint_result, N);
}


//3. LITE's Matrix Multiplication
// CUDA kernel for matrix multiplication
__global__ void matrixMultiplication(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_size = 2;

    // Allocate shared memory for tiles
    __shared__ float As[tile_size][tile_size];
    __shared__ float Bs[tile_size][tile_size];

    float Cvalue = 0.0f;

    for (int k = 0; k < N / tile_size; ++k){
        // Load tiles into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (k * tile_size + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k * tile_size + threadIdx.y) * N + col];

        // Synchronize threads to ensure all data is loaded
        __syncthreads();

        // Perform tile-wise matrix multiplication
        for (int i = 0; i < tile_size; ++i){
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize threads to ensure all data is used before loading the next tiles
        __syncthreads();
    }

    // Store the result in global memory
    C[row * N + col] = Cvalue;
}