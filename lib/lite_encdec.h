#ifndef _LITE_ENCDEC_H_
#define _LITE_ENCDEC_H_

#include "lite.h"

// Forward declarations of AES functions
__device__ void AES_encrypt_gpu(uint *ct, const uint *pt, uint *rek, uint Nr);
__device__ void AES_decrypt_gpu(uint *pt, const uint *ct, uint *rek, uint Nr);
void AES_encrypt_cpu(uint *ct, const uint *pt, uint *rek, uint Nr);
void AES_decrypt_cpu(uint *pt, const uint *ct, uint *rek, uint Nr);

// Encrypt-Decrypt Function declarations
__global__ void ltEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N);

__global__ void ltDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N);

__global__ void ltEncryptGPU(uint *ct, float *pt, uint *rek, uint Nr, int N);

__global__ void ltDecryptGPU(float *pt, uint *ct, uint *rek, uint Nr, int N);

void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N);

void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N);

void ltEncryptCPU(uint *ct, const float *pt, uint *rek, uint Nr, int N);

void ltDecryptCPU(float *pt, const uint *ct, uint *rek, uint Nr, int N);

#endif // _LITE_ENCDEC_H_