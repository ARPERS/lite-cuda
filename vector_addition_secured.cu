#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include "AES/AES_encrypt_cpu.cpp"
#include "AES/AES_encrypt_gpu.cu"
#include "AES/AES_decrypt_cpu.cpp"
#include "AES/AES_decrypt_gpu.cu"
#include "AES/AES.cu"

using namespace std;

__global__ void helloWorld(){
    printf("Halo from GPU!\n");
}

__global__ void vectorAddition(uint *result, uint *a, uint *b, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}

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


__global__ void AESEncryptGPU(const uint *pt, uint *ct, uint *rek, uint Nr){
      AES_encrypt_gpu(pt, ct, rek, Nr);
}

void AESEncryptCPU(const uint *pt, uint *ct, uint *rek, uint Nr){
    AES_encrypt_cpu(pt, ct, rek, Nr);
}

void AESDecryptCPU(const uint *pt, uint *ct, uint *rek, uint Nr){
      AES_decrypt_cpu(pt, ct, rek, Nr);
}

__global__ void AESDecryptGPU(const uint *ct, uint *pt, uint *rek, uint Nr){
      AES_decrypt_gpu(ct, pt, rek, Nr);
}

__global__ void setVal(uint *arr, int i, uint v){
    arr[i] = v;
}

void check(uint target, uint *array, int N){
    bool flag = false;
    for(int i = 0; i < N; i++){
        if(array[i] != target){
            flag = true;
        }
    }
    if(!flag){
        printf("SUCCESS\n");
    }else{
        printf("FAIL\n");
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    int N = 4; // vector length
    size_t size = N * sizeof(uint);

    uint *a = new uint[N];
    uint *b = new uint[N];
    uint *c = new uint[N];
    uint *enc_a = new uint[N];
    uint *enc_b = new uint[N];
    uint *enc_c = new uint[N];

    uint *d_enc_a, *d_enc_b, *d_enc_c;
    uint *d_a, *d_b, *d_c;
    gpuErrchk( cudaMalloc(&d_a, size) );
    gpuErrchk( cudaMalloc(&d_b, size) );
    gpuErrchk( cudaMalloc(&d_c, size) );
    gpuErrchk( cudaMalloc(&d_enc_a, size) );
    gpuErrchk( cudaMalloc(&d_enc_b, size) );
    gpuErrchk( cudaMalloc(&d_enc_c, size) );

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);
    uint *d_e_sched;
    uint *d_d_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_e_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_d_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_e_sched, e_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_d_sched, d_sched, key_size, cudaMemcpyHostToDevice) );

    // initiate
    for(int i = 0; i < N; ++i) {
        a[i] = 35;
        b[i] = 25;
    }

    AESEncryptCPU(a, enc_a, e_sched, Nr);
    AESEncryptCPU(b, enc_b, e_sched, Nr);

    cout << a[0] << " " << a[1] << " " << enc_a[0] << " " << enc_a[1] << endl;
    cout << b[0] << " " << b[1] << " " << enc_b[0] << " " << enc_b[1] << endl;

            // <<<<<<<<<<<<<
    gpuErrchk( cudaMemcpy(d_enc_a, enc_a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_enc_b, enc_b, size, cudaMemcpyHostToDevice) );

    // cudaMemcpy(c, d_enc_b, size, cudaMemcpyDeviceToHost);
    // cout << c[0] << "~" << c[1]  << endl;
                            //>>>>>>>
    AESDecryptGPU<<<1,1>>>(d_enc_a, d_a, d_d_sched, Nr); 
    AESDecryptGPU<<<1,1>>>(d_enc_b, d_b, d_d_sched, Nr);

    // cudaMemcpy(c, d_enc_b, size, cudaMemcpyDeviceToHost);
    // cout << c[0] << "~" << c[1]  << endl;
    // cudaDeviceSynchronize();

    int threadsPerBlock;
    int numberOfBlocks;
    threadsPerBlock = 2;
    numberOfBlocks = (N + threadsPerBlock -  1) / threadsPerBlock;
    vectorAddition<<<numberOfBlocks, threadsPerBlock>>>(d_c, d_a, d_b, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    AESEncryptGPU<<< 1, 1 >>>(d_c, d_enc_c, d_e_sched, Nr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(enc_c, d_enc_c, size, cudaMemcpyDeviceToHost) );

    AESDecryptCPU(enc_c, c, d_sched, Nr);

    cout << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << endl;
    

    check(60, c, N);

    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
}
