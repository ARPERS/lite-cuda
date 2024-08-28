#ifndef _LITE_VECTOR_H_
#define _LITE_VECTOR_H_

#include "lite.h"

#define GRIDSIZE 256
#define BLOCKSIZE 128
// to avoid splling data to global memory
// we only process 512 elements per each block in shared memory
#define BUFFSIZE 512

// Forward declarations of AES functions
__device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
__device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// Vector processing kernel
__global__ void vectorProc(uint *d_enc_result, uint *d_enc_a, uint *d_enc_b, int N,
                           uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float,
                           uint procType=0);

// Wrapper functions for vector processing
void ltVectorProc(uint *result, uint *a, uint *b, int N,
                  uint *enc_sched, uint *dec_sched, int Nr, bool is_float, int procType=0,
                  int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteAddition(uint *result, uint *a, uint *b, int N,
                  uint *enc_sched, uint *dec_sched, int Nr,
                  int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteAddition(float *result, float *a, float *b, int N,
                  uint *enc_sched, uint *dec_sched, int Nr,
                  int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteMultiplication(uint *result, uint *a, uint *b, int N,
                        uint *enc_sched, uint *dec_sched, int Nr,
                        int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteMultiplication(float *result, float *a, float *b, int N,
                        uint *enc_sched, uint *dec_sched, int Nr,
                        int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteSubstraction(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteSubstraction(float *result, float *a, float *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteDivision(uint *result, uint *a, uint *b, int N,
                  uint *enc_sched, uint *dec_sched, int Nr,
                  int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

void liteDivision(float *result, float *a, float *b, int N,
                  uint *enc_sched, uint *dec_sched, int Nr,
                  int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE);

#endif // _LITE_VECTOR_H_
