//  USED FOR BENCHMARKING

#define GRIDSIZE 256
#define BLOCKSIZE 128
#define BUFFSIZE 128 // to avoid splling data to global memory we only process 128 elements per each block

///////////////////////////////////////
// MAIN Lite's Vector-Vector Processing
///////////////////////////////////////
__global__ void vectorProc(uint *d_result, uint *d_a, uint *d_b, int N,
                           bool is_float, uint procType=0){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    if(index < N){
        // __shared__ uint d_a[BUFFSIZE]; ### UNSECURE
        // __shared__ uint d_b[BUFFSIZE];
        // __shared__ uint d_result[BUFFSIZE];
        float d_f_a;
        float d_f_b;
        float d_f_result;

        for(int idx = index; idx < N; idx += stride){
            // printf("%d %d %d %d %d %d\n", threadIdx.x, blockIdx.x, index, stride, idx, idx*4+3);

            // GPU Decrypt ### UNSECURE
            // if(threadIdx.x%4==0){
            //     AES_decrypt_gpu(d_a + (idx % BUFFSIZE), d_enc_a + idx, d_dec_sched, Nr); 
            //     AES_decrypt_gpu(d_b + (idx % BUFFSIZE), d_enc_b + idx, d_dec_sched, Nr);  
            // }
            // __syncthreads();

            if(is_float){
                d_f_a = *uintToFloat(&d_a[idx]);
                d_f_b = *uintToFloat(&d_b[idx]);
                if(procType==0)
                    d_f_result = d_f_a + d_f_b;
                d_result[idx] = *floatToUint(&d_f_result);
            }else{
                if(procType==0)
                    d_result[idx] = d_a[idx] + d_b[idx];
            }
            
            // GPU Encrypt ### UNSECURE
            // if(threadIdx.x%4==0){
            //     AES_encrypt_gpu(d_enc_result + idx, d_result + (idx % BUFFSIZE), d_enc_sched, Nr);
            // }
        }
    }
}
// wrapper vector processing for CPU-GPU comm.
void ltVectorProc(uint *result, uint *a, uint *b, int N,
                    bool is_float, int procType=0,
                    int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE){
    // Check size, pad so it's divisible by 4
    int padSizeA = padArray(a, N);
    int padSizeB = padArray(b, N);
    
    N += padSizeA; // assuming the size is the same
    
    // CPU Encrypt N elements ### UNSECURE
    // uint *enc_a = new uint[N];
    // uint *enc_b = new uint[N];
    // uint *enc_result = new uint[N];
    // ltEncryptCPU(enc_a, a, enc_sched, Nr, N);
    // ltEncryptCPU(enc_b, b, enc_sched, Nr, N);

    // CPU -> GPU: Data
    uint *d_a, *d_b, *d_result;
    size_t size = sizeof(uint)*N;
    gpuErrchk( cudaMalloc(&d_a, size) );
    gpuErrchk( cudaMalloc(&d_b, size) );
    gpuErrchk( cudaMalloc(&d_result, size) );
    gpuErrchk( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) );

    // CPU -> GPU: Key ### UNSECURE
    // uint *d_enc_sched;
    // uint *d_dec_sched;
    // size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    // gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    // gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    // gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
    // gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );
    
    vectorProc<<<gridSize, blockSize>>>(d_result, d_a, d_b, N, is_float, procType);

    // printf("----Leak Global Memory of The Result-------\n");
    // float *tmp = new float[N];
    // for(int i=0;i<N;i++) if(is_float) memcpy(&tmp[i], &result[i], sizeof(uint)),  printf("%.4f ", tmp[i]); else printf("%u ",result[i]); printf("\n");
    // printf("-------------------------------------------\n");

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost) );

    // CPU Decrypt  ### UNSECURE
    // ltDecryptCPU(result, enc_result, dec_sched, Nr, N);

    removePadArray(a, N, padSizeA);
    removePadArray(b, N, padSizeB);
    N -= padSizeA;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

/////// ADDITION
// front-end wrapper vector addtion for uint array
void liteAddition(uint *result, uint *a, uint *b, int N,
                      int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE){
   ltVectorProc(result, a, b, N, false, 0, gridSize, blockSize);
}
// front-end wrapper vector addtion for float array
void liteAddition(float *result, float *a, float *b, int N,
                      int gridSize=GRIDSIZE, int blockSize=BLOCKSIZE){
    // Float array to uint array
    uint *uint_a = new uint[N];
    uint *uint_b = new uint[N];
    uint *uint_result = new uint[N];
    floatToUintCPU(uint_a, a, N);
    floatToUintCPU(uint_b, b, N);
    
    ltVectorProc(uint_result, uint_a, uint_b, N, true, 0, gridSize, blockSize);

    // uint to float
    uintToFloatCPU(result, uint_result, N);
}