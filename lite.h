#ifndef _LITE_H_
#define _LITE_H_

#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cstring>

#ifndef _AES_SCRIPTS_H_
#define _AES_SCRIPTS_H_
#include "AES/AES.cu"
#include "AES/AES_encrypt_cpu.cpp"
#include "AES/AES_encrypt_gpu.cu"
#include "AES/AES_decrypt_cpu.cpp" 
#include "AES/AES_decrypt_gpu.cu"
#endif

#include "lite_utils.h"
#include "lite_encdec.h"
#include "lite_vector.h"
#include "lite_matrix.h"

// LITE UTILS 

// Debugging macro
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
//    if (code != cudaSuccess){
//       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//       if (abort) exit(code);
//    }
// }

// // Device functions
// __device__ unsigned int* floatToUint(float *input);
// __device__ float* uintToFloat(unsigned int *input);

// // Host functions
// void floatToUintCPU(uint *dest, const float *source, int N);
// void uintToFloatCPU(float *dest, const uint *source, int N);
// int padArray(uint* arr, int N);
// void removePadArray(uint* arr, int N, int paddingSize);
// int padMatrix(uint* matrix, int& width, int& height);
// void removePadMatrix(int* matrix, int& width, int& height, int paddingSize);



// // LITE ENCDEC
// // Forward declarations of AES functions
// __device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
// __device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);
// void AES_encrypt_cpu(uint *ct, const uint *pt, uint *rek, uint Nr);
// void AES_decrypt_cpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// // Encrypt-Decrypt Function declarations
// __global__ void ltEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N);

// __global__ void ltDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N);

// __global__ void ltEncryptGPU(uint *ct, float *pt, uint *rek, uint Nr, int N);

// __global__ void ltDecryptGPU(float *pt, uint *ct, uint *rek, uint Nr, int N);

// void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N);

// void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N);

// void ltEncryptCPU(uint *ct, const float *pt, uint *rek, uint Nr, int N);

// void ltDecryptCPU(float *pt, const uint *ct, uint *rek, uint Nr, int N);

// // LITE MATRIX
// #define TILE_SIZE 4
// // Forward declarations of AES functions
// __device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
// __device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// // Matrix multiplication kernel
// __global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N,
//                                      uint *d_enc_sched, uint *d_dec_sched, int Nr,
//                                      bool is_float, bool is_secure);

// // Wrapper functions for matrix multiplication
// void ltMatMultiplication(uint *result, uint *A, uint *B, int N,
//                          uint *enc_sched, uint *dec_sched, int Nr,
//                          bool is_float, bool is_secure);

// void liteMatMultiplication(uint *result, uint *A, uint *B, int N,
//                            uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true);

// void liteMatMultiplication(float *result, float *A, float *B, int N,
//                            uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true);


// // LITE VECTOR

// #define GRIDSIZE 256
// #define BLOCKSIZE 128
// // to avoid splling data to global memory
// // we only process 512 elements per each block in shared memory
// #define BUFFSIZE 512

// // Forward declarations of AES functions
// __device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
// __device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// // Vector processing kernel
// __global__ void vectorProc(uint *d_enc_result, uint *d_enc_a, uint *d_enc_b, int N,
//                            uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float,
//                            uint procType=0);

// // Wrapper functions for vector processing
// void ltVectorProc(uint *result, uint *a, uint *b, int N,
//                   uint *enc_sched, uint *dec_sched, int Nr, bool is_float, int procType=0,
//                   int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteAddition(uint *result, uint *a, uint *b, int N,
//                   uint *enc_sched, uint *dec_sched, int Nr,
//                   int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteAddition(float *result, float *a, float *b, int N,
//                   uint *enc_sched, uint *dec_sched, int Nr,
//                   int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteMultiplication(uint *result, uint *a, uint *b, int N,
//                         uint *enc_sched, uint *dec_sched, int Nr,
//                         int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteMultiplication(float *result, float *a, float *b, int N,
//                         uint *enc_sched, uint *dec_sched, int Nr,
//                         int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteSubstraction(uint *result, uint *a, uint *b, int N,
//                       uint *enc_sched, uint *dec_sched, int Nr,
//                       int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteSubstraction(float *result, float *a, float *b, int N,
//                       uint *enc_sched, uint *dec_sched, int Nr,
//                       int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteDivision(uint *result, uint *a, uint *b, int N,
//                   uint *enc_sched, uint *dec_sched, int Nr,
//                   int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

// void liteDivision(float *result, float *a, float *b, int N,
//                   uint *enc_sched, uint *dec_sched, int Nr,
//                   int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);


#endif // LITE_H
