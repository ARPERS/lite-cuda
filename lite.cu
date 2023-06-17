#include <stdio.h>
#include <iostream>
#include <typeinfo>

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
__global__ void setVal(uint *arr, int i, uint v){
    arr[i] = v;
}

//0. Utils
uint32_t floatToUInt(float value) {
  union {
    float floatValue;
    uint32_t uintValue;
  } u;
  u.floatValue = value;
  return u.uintValue;
}

float uintToFloat(uint32_t value) {
  union {
    float floatValue;
    uint32_t uintValue;
  } u;
  u.uintValue = value;
  return u.floatValue;
}

//1. Encrypt-Decrypt Function
__global__ void AESEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr){
      AES_encrypt_gpu(ct, pt, rek, Nr);
}
void AESEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr){
    AES_encrypt_cpu(ct, pt, rek, Nr);
}
void AESDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr){
      AES_decrypt_cpu(pt, ct, rek, Nr);
}
__global__ void AESDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr){
      AES_decrypt_gpu(pt, ct, rek, Nr);
}

//2. Lite's Vector Addition
__global__ void ltVectorAddition(uint *result, uint *a, uint *b, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}
__global__ void ltVectorAddition(float *result, float *a, float *b, int N){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    memcpy(uint_a, a, sizeof(a));
    memcpy(uint_b, b, sizeof(b));
    memcpy(uint_result, result, sizeof(result));
    
    // uint to float
    memcpy(a, uint_a, sizeof(uint_a));
    memcpy(b, uint_b, sizeof(uint_b));
    memcpy(result, uint_result, sizeof(uint_result));
}




