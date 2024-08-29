/*
Example for encrypting and decrypting 8 float values using LITE
*/

#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include <lite.cu>

using namespace std;

int main() {
    int N = 8; // vector length

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
    float *x = (float*)malloc(bytes);  // original value
    uint *y = (uint*)malloc(bytes);  // to store encrypted x in CPU
    float *z = (float*)malloc(bytes);  // to store decrypted y in CPU
    x[0] = 12.3; x[1] = 2.22; x[2]=9.89; x[3]=27.5; 
    x[4] = 912.3; x[5] = 92.22; x[6]=99.89; x[7]=927.5; 

    // CPU Encrypt
    cout << "CPU Pln Text: "; for(int i=0;i<N;i++) cout << x[i] << " "; cout << endl;
    ltEncryptCPU(y, x, e_sched, Nr, N);
      
    // CPU Decrypt
    cout << "CPU Chp Text: "; for(int i=0;i<N;i++) cout << y[i] << " "; cout << endl;
    ltDecryptCPU(z, y, d_sched, Nr, N);
    cout << "CPU Pln Text: "; for(int i=0;i<N;i++) cout << z[i] << " "; cout << endl;

    cout << "-----------\n";

    // Initiating values in GPU
    float *d_x,*d_z;
    uint *d_y; 
    gpuErrchk( cudaMalloc(&d_x, bytes) );  // original value
    gpuErrchk( cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMalloc(&d_y, bytes) );  // to store encrypted x in GPU 
    gpuErrchk( cudaMalloc(&d_z, bytes) );  // to store decrypted y in GPU

    // Send Key to GPU
    uint *d_e_sched;
    uint *d_d_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_e_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_d_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_e_sched, e_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_d_sched, d_sched, key_size, cudaMemcpyHostToDevice) );

    cudaMemcpy(x, d_x, bytes, cudaMemcpyDeviceToHost);
    cout << "GPU Pln Text: "; for(int i=0;i<N;i++) cout << x[i] << " "; cout << endl;

    // GPU Encrypt
    ltEncryptGPU<<< 1, N/4 >>>(d_y, d_x, d_e_sched, Nr, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost) );
    cout << "GPU Chp Text: "; for(int i=0;i<N;i++) cout << y[i] << " "; cout << endl;
    
    // GPU Decrypt
    ltDecryptGPU<<<1,N/4>>>(d_z, d_y, d_d_sched, Nr, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(z, d_z, bytes, cudaMemcpyDeviceToHost) );
    cout << "GPU Pln Text: "; for(int i=0;i<N;i++) cout << z[i] << " "; cout << endl;
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
}
