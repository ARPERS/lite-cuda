#include "lite_encdec.h"

///////////////////////////////////////
//1. Encrypt-Decrypt Function
///////////////////////////////////////
__global__ void ltEncryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N){ // global handle N encryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index*4 + 3 < N)
        AES_encrypt_gpu(ct+index*4, pt+index*4, rek, Nr); // device handle 128 bit encryption
}
__global__ void ltDecryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N){ // global handle N decryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index*4 + 3 < N)
        AES_decrypt_gpu(pt+index*4, ct+index*4, rek, Nr); // device handle 128 bit encryption
}
__global__ void ltEncryptGPU(uint *ct, float *pt, uint *rek, uint Nr, int N){ // global handle N encryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint f_pt[4];
    for(int i=0; i<4; i++){
        f_pt[i] = *floatToUint(&pt[index*4+i]);
    }
    if(index*4 + 3 < N)
        AES_encrypt_gpu(ct+index*4, f_pt, rek, Nr); // device handle 128 bit encryption
}
__global__ void ltDecryptGPU(float *pt, uint *ct, uint *rek, uint Nr, int N){ // global handle N decryption
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uint f_pt[4];
    if(index*4 + 3 < N)
        AES_decrypt_gpu(f_pt, ct+index*4, rek, Nr); // device handle 128 bit encryption
    for(int i=0;i<4;i++){
        pt[index*4+i] = *uintToFloat(&f_pt[i]);
    }
}
void ltEncryptCPU(uint *ct, const uint *pt, uint *rek, uint Nr, int N){ // run encrypt for all elements
    for(int i=0;i<N;i+=4){
      AES_encrypt_cpu(ct+i, pt+i, rek, Nr);
    }
}
void ltDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr, int N){ // run decrypt for all elements
    for(int i=0;i<N;i+=4){
      AES_decrypt_cpu(pt+i, ct+i, rek, Nr);
    }
}
void ltEncryptCPU(uint *ct, const float *pt, uint *rek, uint Nr, int N){ // run encrypt for all elements
    uint *uint_pt = new uint[N];
    floatToUintCPU(uint_pt, pt, N);
    for(int i=0;i<N;i+=4){
      AES_encrypt_cpu(ct+i, uint_pt+i, rek, Nr);
    }
}
void ltDecryptCPU(float *pt, const uint *ct, uint *rek, uint Nr, int N){ // run decrypt for all elements
    uint *uint_pt = new uint[N];
    for(int i=0;i<N;i+=4){
      AES_decrypt_cpu(uint_pt+i, ct+i, rek, Nr);
    }
    uintToFloatCPU(pt, uint_pt, N);
}