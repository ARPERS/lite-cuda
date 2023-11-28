// USED FOR BENCHMARKING

///////////////////////////////////////
//   MAIN LITE's MATRIX Multiplication
///////////////////////////////////////
__global__ void matrixMultiplication(uint *C, uint *A, uint *B, int N,
                                     uint *d_enc_sched, uint *d_dec_sched, int Nr,
                                     bool is_float, bool is_secure){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_size = 4;

    // Allocate shared memory for tiles
    __shared__ uint As[tile_size][tile_size];
    __shared__ uint Bs[tile_size][tile_size];
    __shared__ uint Cs[tile_size][tile_size];

    uint tempTotalUint = 0;
    float tempTotalFloat = 0.0f;

    for (int k = 0; k < N / tile_size; ++k){
        // Load tiles into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + (k * tile_size + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k * tile_size + threadIdx.y) * N + col];
    
        // To ensure all data is loaded
        __syncthreads();

        if(is_secure){
            AES_decrypt_gpu(As[threadIdx.y], As[threadIdx.y], d_dec_sched, Nr); 
            AES_decrypt_gpu(Bs[threadIdx.y], Bs[threadIdx.y], d_dec_sched, Nr); 
            // To ensure all data is decrypted
            __syncthreads();
        }
    
        // Tile-wise matrix multiplication
        for (int i = 0; i < tile_size; ++i){
            if(is_float){
                tempTotalFloat += *uintToFloat(&As[threadIdx.y][i]) * *uintToFloat(&Bs[i][threadIdx.x]);
            }else{
                tempTotalUint += As[threadIdx.y][i] * Bs[i][threadIdx.x];
            }
        }
        // To ensure all data is used before loading the next tiles
        __syncthreads();
    }

    if(is_float){
        Cs[threadIdx.y][threadIdx.x] = *floatToUint(&tempTotalFloat);
    }else{
        Cs[threadIdx.y][threadIdx.x] = tempTotalUint;
    }
    
   
    if(is_secure){
        AES_encrypt_gpu(Cs[threadIdx.y], Cs[threadIdx.y], d_enc_sched, Nr); 
         // To ensure all data is encrypted
        __syncthreads();
    }

    // Store the encrypted result in global memory
    C[row * N + col] = Cs[threadIdx.y][threadIdx.x];
}
// wrapper matrix multiplication for CPU-GPU comm.
void ltMatMultiplication(uint *result, uint *A, uint *B, int N,
                            uint *enc_sched, uint *dec_sched, int Nr,
                            bool is_float, bool is_secure){
    const int TILE_SIZE = 4;

    uint *d_a, *d_b, *d_result;
    uint *d_enc_sched; // for secure key
    uint *d_dec_sched; // for secure key
    size_t size = sizeof(uint)*N*N;
    gpuErrchk( cudaMalloc(&d_a, size) );
    gpuErrchk( cudaMalloc(&d_b, size) );
    gpuErrchk( cudaMalloc(&d_result, size) );

    if(is_secure){
        uint *enc_a = new uint[N*N];
        uint *enc_b = new uint[N*N];

        // CPU Encrypt NxN elements
        ltEncryptCPU(enc_a, A, enc_sched, Nr, N*N); 
        ltEncryptCPU(enc_b, B, enc_sched, Nr, N*N);

        // CPU -> GPU: Data
        gpuErrchk( cudaMemcpy(d_a, enc_a, size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_b, enc_b, size, cudaMemcpyHostToDevice) );

        // uint *tmp = new uint[N*N];  // debugger
        // gpuErrchk( cudaMemcpy(tmp, d_enc_a, size, cudaMemcpyDeviceToHost) ); 
        // printf("----A FROM GPU------------\n");
        // for(int i=0;i<N*N;i++) printf("%u ", tmp[i]); printf("\n");
        // printf("----------------------------\n");

        // CPU -> GPU: Key
        size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
        gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
        gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
        gpuErrchk( cudaMemcpy(d_enc_sched, enc_sched, key_size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_dec_sched, dec_sched, key_size, cudaMemcpyHostToDevice) );
    }else{
        // CPU -> GPU: Data UNSECURE
        gpuErrchk( cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice) );
    }

    // Define grid and block dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize(N / TILE_SIZE, N / TILE_SIZE);

    matrixMultiplication<<<gridSize, blockSize>>>(d_result, d_a, d_b, N, d_enc_sched, d_dec_sched, Nr, is_float, is_secure);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // GPU -> CPU
    gpuErrchk( cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost) );

    // printf("----Leak Global Memory of The Result-------\n");
    // float *tmp = new float[N*N];
    // for(int i=0;i<N*N;i++) if(is_float) memcpy(&tmp[i], &result[i], sizeof(uint)),  printf("%.4f ", tmp[i]); else printf("%u ",result[i]); printf("\n");
    // printf("-------------------------------------------\n");

    if(is_secure){
        // CPU Decrypt
        ltDecryptCPU(result, result, dec_sched, Nr, N*N);
    }
}
// wrapper matrix multiplication for uint matrix
void liteMatMultiplication(uint *result, uint *A, uint *B, int N,
                            uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true){
    ltMatMultiplication(result, A, B, N, enc_sched, dec_sched, Nr, false, is_secure);
}
// wrapper matrix multiplication for float matrix
void liteMatMultiplication(float *result, float *A, float *B, int N,
                            uint *enc_sched, uint *dec_sched, int Nr, bool is_secure = true){
    // Float array to uint array
    uint *uint_a = new uint[N*N];
    uint *uint_b = new uint[N*N];
    uint *uint_result = new uint[N*N];

    floatToUintCPU(uint_a, A, N*N);
    floatToUintCPU(uint_b, B, N*N);

    ltMatMultiplication(uint_result, uint_a, uint_b, N, enc_sched, dec_sched, Nr, true, is_secure);
    
    // uint to float
    uintToFloatCPU(result, uint_result, N*N);
}