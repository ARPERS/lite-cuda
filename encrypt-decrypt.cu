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

__global__ void AESEncryptGPU(uint *pt, const uint *ct, uint *rek, uint Nr){
      AES_encrypt_gpu(pt, ct, rek, Nr);
}

void AESEncryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr){
    AES_encrypt_cpu(pt, ct, rek, Nr);
}

void AESDecryptCPU(uint *pt, const uint *ct, uint *rek, uint Nr){
      AES_decrypt_cpu(pt, ct, rek, Nr);
}

__global__ void AESDecryptGPU(uint *ct, const uint *pt, uint *rek, uint Nr){
      AES_decrypt_gpu(ct, pt, rek, Nr);
}

__global__ void setVal(uint *arr, int i, uint v){
    arr[i] = v;
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

    // key declaration
    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    // Initiating values in CPU
    size_t bytes = N * sizeof(uint);
    uint *x = (uint*)malloc(bytes);
    uint *y = (uint*)malloc(bytes);
    uint *z = (uint*)malloc(bytes);
    x[0] = 123; x[1] = 222; x[2]=989; x[3]=275; 

    // Send Key to GPU
    uint *d_e_sched;
    uint *d_d_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_e_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_d_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_e_sched, e_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_d_sched, d_sched, key_size, cudaMemcpyHostToDevice) );

    // CPU
    cout << "CPU Pln Text: "<< x[0] << " " << x[1] << " " << x[2] << " " << x[3] << endl;
    AESEncryptCPU(y, x, e_sched, Nr);
    cout << "CPU Pln Text: "<< y[0] << " " << y[1] << " " << y[2] << " " << y[3] << endl;
    AESDecryptCPU(z, y, d_sched, Nr);
    cout << "CPU Pln Text: "<< z[0] << " " << z[1] << " " << z[2] << " " << z[3] << endl;
   
    cout << "-----------\n";

    // Initiating values in GPU
    uint *d_x, *d_y, *d_z;
    gpuErrchk( cudaMalloc(&d_x, bytes) ); 
    gpuErrchk( cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&d_y, bytes) );
    gpuErrchk( cudaMalloc(&d_z, bytes) );

    cudaMemcpy(x, d_x, bytes, cudaMemcpyDeviceToHost);
    cout << "GPU Pln Text: " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << endl;

    AESEncryptGPU<<< 1, 1 >>>(d_y, d_x, d_e_sched, Nr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
   
    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);
    cout << "GPU Chp Text: " << y[0] << " " << y[1] << " " << y[2] << " " << y[3] << endl;
    
    AESDecryptGPU<<<1,1>>>(d_z, d_y, d_d_sched, Nr);
    cudaMemcpy(z, d_z, bytes, cudaMemcpyDeviceToHost);
    cout << "GPU Pln Text: " << z[0] << " " << z[1] << " " << z[2] << " " << z[3] << endl;
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
