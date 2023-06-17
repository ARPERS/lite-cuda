#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include "lite.cu"

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

    AESEncryptCPU(enc_a, a, e_sched, Nr);
    AESEncryptCPU(enc_b, b, e_sched, Nr);

    cout << a[0] << " " << a[1] << " " << enc_a[0] << " " << enc_a[1] << endl;
    cout << b[0] << " " << b[1] << " " << enc_b[0] << " " << enc_b[1] << endl;

            // <<<<<<<<<<<<<
    gpuErrchk( cudaMemcpy(d_enc_a, enc_a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_enc_b, enc_b, size, cudaMemcpyHostToDevice) );

    AESDecryptGPU<<<1,1>>>(d_a, d_enc_a, d_d_sched, Nr); 
    AESDecryptGPU<<<1,1>>>(d_b, d_enc_b, d_d_sched, Nr);

    ltVectorAddition<<<1, N>>>(d_c, d_a, d_b, N);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    AESEncryptGPU<<< 1, 1 >>>(d_enc_c, d_c, d_e_sched, Nr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(enc_c, d_enc_c, size, cudaMemcpyDeviceToHost) );

    AESDecryptCPU(c, enc_c, d_sched, Nr);

    cout << c[0] << " " << c[1] << " " << c[2] << " " << c[3] << endl;
    

    check(60, c, N);

    // cudaFree(a);
    // cudaFree(b);
    // cudaFree(c);
}
