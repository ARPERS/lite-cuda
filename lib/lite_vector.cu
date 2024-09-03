#include "lite_vector.h"

///////////////////////////////////////
// MAIN Lite's Vector-Vector Processing
///////////////////////////////////////
__global__ void vectorProc(uint *d_enc_result, uint *d_enc_a, uint *d_enc_b, int N,
                           uint *d_enc_sched, uint *d_dec_sched, int Nr, bool is_float,
                           uint procType){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    if(index*4 < N){
        __shared__ uint d_a[BUFFSIZE];
        __shared__ uint d_b[BUFFSIZE];
        __shared__ uint d_result[BUFFSIZE];
        
        __shared__ float d_f_a[BUFFSIZE];
        __shared__ float d_f_b[BUFFSIZE];
        __shared__ float d_f_result[BUFFSIZE];

        for(int st = index; st < N; st += stride){

            AES_decrypt_gpu(d_a + ((st*4) % BUFFSIZE), d_enc_a + (st*4), d_dec_sched, Nr); 
            AES_decrypt_gpu(d_b + ((st*4) % BUFFSIZE), d_enc_b + (st*4), d_dec_sched, Nr);
            // __syncthreads();

            for(int i=0;i<4;i++){
                int idx = threadIdx.x*4+i;
                if(is_float){
                    d_f_a[idx] = *uintToFloat(&d_a[idx]);
                    d_f_b[idx] = *uintToFloat(&d_b[idx]);
                    if(procType==0)
                        d_f_result[idx] = d_f_a[idx] + d_f_b[idx];
                    else if(procType==1)
                        d_f_result[idx] = d_f_a[idx] * d_f_b[idx];
                    else if(procType==2)
                        d_f_result[idx] = d_f_a[idx] - d_f_b[idx];
                    else if(procType==3)
                        d_f_result[idx] = d_f_a[idx] / d_f_b[idx];
                    d_result[idx] = *floatToUint(&d_f_result[idx]);
                }else{
                    if(procType==0)
                        d_result[idx] = d_a[idx] + d_b[idx];
                    else if(procType==1)
                        d_result[idx] = d_a[idx] * d_b[idx];
                    else if(procType==2)
                        d_result[idx] = d_a[idx] - d_b[idx];
                    else if(procType==3)
                        d_result[idx] = d_a[idx] / d_b[idx];
                }
            }

            // __syncthreads();
            AES_encrypt_gpu(d_enc_result + (st*4), d_result + ((st*4) % BUFFSIZE), d_enc_sched, Nr);
        }
    }
}
// wrapper vector processing for CPU-GPU comm.
void ltVectorProc(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr, bool is_float, int procType,
                      int gridSize, int blockSize){
    // Check size, pad so it's divisible by 4
    int padSizeA = padArray(a, N);
    int padSizeB = padArray(b, N);
    
    N += padSizeA; // assuming the size is the same
    
    // CPU Encrypt N elements
    uint *enc_a = new uint[N];
    uint *enc_b = new uint[N];
    uint *enc_result = new uint[N];
    ltEncryptCPU(enc_a, a, enc_sched, Nr, N);
    ltEncryptCPU(enc_b, b, enc_sched, Nr, N);

    // CPU -> GPU: Data
    uint *d_enc_a, *d_enc_b, *d_enc_result;
    size_t size = sizeof(uint)*N;
    gpuErrchk( cudaMalloc(&d_enc_a, size) );
    gpuErrchk( cudaMalloc(&d_enc_b, size) );
    gpuErrchk( cudaMalloc(&d_enc_result, size) );
    gpuErrchk( cudaMemcpy(d_enc_a, enc_a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_enc_b, enc_b, size, cudaMemcpyHostToDevice) );

    // CPU -> GPU: Key
    uint *d_enc_sched;
    uint *d_dec_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );
    
    vectorProc<<<gridSize, blockSize>>>(d_enc_result, d_enc_a, d_enc_b, N,d_enc_sched, d_dec_sched, Nr, is_float, procType);

    // printf("----Leak Global Memory of The Result-------\n");
    // float *tmp = new float[N];
    // for(int i=0;i<N;i++) if(is_float) memcpy(&tmp[i], &result[i], sizeof(uint)),  printf("%.4f ", tmp[i]); else printf("%u ",result[i]); printf("\n");
    // printf("-------------------------------------------\n");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(enc_result, d_enc_result, size, cudaMemcpyDeviceToHost) );

    // CPU Decrypt
    ltDecryptCPU(result, enc_result, dec_sched, Nr, N);

    removePadArray(a, N, padSizeA);
    removePadArray(b, N, padSizeB);
    N -= padSizeA;
    
    cudaFree(d_enc_a);
    cudaFree(d_enc_b);
    cudaFree(d_enc_sched);
    cudaFree(d_dec_sched);
}

/////// ADDITION
// front-end wrapper vector addtion for uint array
void ltVecAdd(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
   ltVectorProc(result, a, b, N, enc_sched, dec_sched, Nr, false, 0, gridSize, blockSize);
}
// front-end wrapper vector addtion for float array
void ltVecAdd(float *result, float *a, float *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);

    ltVectorProc(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, 0, gridSize, blockSize);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}

/////// MULTIPLICATION
// front-end wrapper vector multiplication for uint array
void ltVecMul(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
   ltVectorProc(result, a, b, N, enc_sched, dec_sched, Nr, false, 1, gridSize, blockSize);
}
// front-end wrapper vector multiplication for float array
void ltVecMul(float *result, float *a, float *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);

    ltVectorProc(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, 1, gridSize, blockSize);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}
// front-end wrapper vector substraction for uint array
void ltVecSub(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
   ltVectorProc(result, a, b, N, enc_sched, dec_sched, Nr, false, 2, gridSize, blockSize);
}

/////// SUBSTRACTION
// front-end wrapper vector addtion for float array
void ltVecSub(float *result, float *a, float *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);

    ltVectorProc(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, 2, gridSize, blockSize);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}

/////// DIVISION
// front-end wrapper vector division for uint array
void ltVecDiv(uint *result, uint *a, uint *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
   ltVectorProc(result, a, b, N, enc_sched, dec_sched, Nr, false, 3, gridSize, blockSize);
}
// front-end wrapper vector addtion for float array
void ltVecDiv(float *result, float *a, float *b, int N,
                      uint *enc_sched, uint *dec_sched, int Nr,
                      int gridSize, int blockSize){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);

    ltVectorProc(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, 3, gridSize, blockSize);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}