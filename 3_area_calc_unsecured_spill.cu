/*
This code try to implement LITE's step by step
but accidentally spill the data to GPU memory
*/
#include <iostream>
#include <typeinfo>
#include <cmath>

#include <cuda.h>
#include "lite.cu"


#include <chrono>

__global__ void calculatePolygonArea(float* area, float* x, float* y, int n_points) {
    __shared__ float sharedArea[1];
    sharedArea[0] = 0.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int next_idx = (idx + 1) % n_points;

    float term = (x[idx] * y[next_idx] - x[next_idx] * y[idx]);
    
    atomicAdd(sharedArea, term);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(area, 0.5f * sharedArea[0]);
    }
}

int main() {

    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double avg_time = 0; 
    auto t1 = high_resolution_clock::now();

    int n_points = 8;
    float h_f_x[n_points]; // Host arrays for Float x-coordinates
    float h_f_y[n_points]; // Host arrays for Float y-coordinates


    // Define the points of a quadrilateral
    h_f_x[0] = 6.0f; h_f_y[0] = 4.0f;
    h_f_x[1] = 6.0f; h_f_y[1] = 12.0f;
    h_f_x[2] = 11.0f; h_f_y[2] = 12.0f;
    h_f_x[3] = 14.0f; h_f_y[3] = 6.0f;
    h_f_x[4] = 12.0f; h_f_y[4] = 0.0f;
    h_f_x[5] = 12.0f; h_f_y[5] = -3.0f;
    h_f_x[6] = 10.0f; h_f_y[6] = -5.0f;
    h_f_x[7] = 4.0f; h_f_y[7] = -2.0f;
    
    // 0. Initiate key in CPU
    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    // 0. Move Key to GPU
    uint *d_enc_sched;
    uint *d_dec_sched;
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_enc_sched, e_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dec_sched, d_sched, key_size, cudaMemcpyHostToDevice) );
    
    
    // 1. Encrypt FLOAT to INT in CPU
    uint* h_enc_i_x = new uint[n_points];
    uint* h_enc_i_y = new uint[n_points];
    ltEncryptCPU(h_enc_i_x, h_f_x, e_sched, Nr, n_points);
    ltEncryptCPU(h_enc_i_y, h_f_y, e_sched, Nr, n_points);

    // 1. Copy data from host to device
    uint* d_enc_i_x;     // Device encrypted int array to receive x-coordinates
    uint* d_enc_i_y;     // Device encrypted int array to receive y-coordinates
    cudaMalloc((void**)&d_enc_i_x, sizeof(uint) * n_points);
    cudaMalloc((void**)&d_enc_i_y, sizeof(uint) * n_points);
    cudaMemcpy(d_enc_i_x, h_enc_i_x, sizeof(uint) * n_points, cudaMemcpyHostToDevice);
    cudaMemcpy(d_enc_i_y, h_enc_i_y, sizeof(uint) * n_points, cudaMemcpyHostToDevice);
    
    // 2. Decrypt INT to FLOAT in GPU  <<<!!SPILLING DECRYPTED ATA TO GPU MEMORY!!>>>                             
    float* d_f_x;     // Device encrypted int array to receive x-coordinates
    float* d_f_y;     // Device encrypted int array to receive y-coordinates
    cudaMalloc((void**)&d_f_x, sizeof(float) * n_points);
    cudaMalloc((void**)&d_f_y, sizeof(float) * n_points);
    ltDecryptGPU<<<1, n_points>>>(d_f_x, d_enc_i_x, d_dec_sched, Nr, n_points);
    ltDecryptGPU<<<1, n_points>>>(d_f_y, d_enc_i_y, d_dec_sched, Nr, n_points);

    // 3. Launch the CUDA kernel get Decrypted Float Area
    float* d_f_area;  // Device variable to store the area
    cudaMalloc((void**)&d_f_area, 4*sizeof(float));
    calculatePolygonArea<<<1, n_points>>>(&d_f_area[0], d_f_x, d_f_y, n_points);

    // 4. Encrypt Result FLOAT to INT in GPU
    uint* d_enc_i_area;  // Device variable to store the area
    cudaMalloc((void**)&d_enc_i_area, 4*sizeof(uint));
    ltEncryptGPU<<<1, 1>>>(d_enc_i_area, d_f_area, d_enc_sched, Nr, 4);

    // 4. Copy the result back from the device
    uint* h_enc_i_area = new uint[4];
    cudaMemcpy(h_enc_i_area, d_enc_i_area, 4*sizeof(uint), cudaMemcpyDeviceToHost);

    // 5. Decrypt the result
    float* h_area = new float[4];        // Host variable to store the area
    ltDecryptCPU(h_area, h_enc_i_area, d_sched, Nr, 4);
    
    auto t2 = high_resolution_clock::now();

    cudaFree(d_enc_sched);
    cudaFree(d_enc_i_x);
    cudaFree(d_enc_i_y);
    cudaFree(d_f_x);
    cudaFree(d_f_y);
    cudaFree(d_f_area);
    cudaFree(d_enc_i_area);

    std::cout << "Area of the polygon: " << h_area[0] << std::endl;

    duration<double, std::milli> ms_double = t2 - t1;
    avg_time += ms_double.count();
    std::cout << avg_time << std::endl;

    return 0;
}
