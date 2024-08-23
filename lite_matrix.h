#ifndef _LITE_MATRIX_H_
#define _LITE_MATRIX_H_

#include "lite.h"

#define TILE_SIZE 4

// Forward declarations of AES functions
__device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
__device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// Matrix multiplication kernel
__global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N,
                                     uint *d_enc_sched, uint *d_dec_sched, int Nr,
                                     bool is_float, bool is_secure);

// Wrapper functions for matrix multiplication
void ltMatMultiplication(uint *result, uint *A, uint *B, int N,
                         uint *enc_sched, uint *dec_sched, int Nr,
                         bool is_float, bool is_secure);

void liteMatMultiplication(uint *result, uint *A, uint *B, int N,
                           uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true);

void liteMatMultiplication(float *result, float *A, float *B, int N,
                           uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true);

#endif // _LITE_MATRIX_H_
