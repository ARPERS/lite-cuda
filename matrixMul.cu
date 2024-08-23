/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>

// includes, project
#ifdef __MCUDA__
#include <mcuda.h>
#endif

#include "matrixMul.h"
#include "lite.cu"
#include "matrixMul_kernel_lite.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void randomInit(uint*, int);
bool checkResult(uint* C, uint* A, uint* B, int wA, int wB, int wC, int hC);

__global__ void matrixMulSecure(uint* C, uint* A, uint* B, int wA, int wB, uint *d_enc_sched, uint *d_dec_sched, int Nr);

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    // set seed for rand()
    srand(2006);

    // @@@ LITE:create the secure key
    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);
    size_t key_size = (4*(MAXNR + 1)) * sizeof(uint);
    uint *d_enc_sched; // for device secure key
    uint *d_dec_sched; // for device secure key
    gpuErrchk( cudaMalloc(&d_enc_sched, key_size) );
    gpuErrchk( cudaMalloc(&d_dec_sched, key_size) );
    gpuErrchk( cudaMemcpy(d_enc_sched, e_sched, key_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_dec_sched, d_sched, key_size, cudaMemcpyHostToDevice) );
    // @@@ END-LITE:create the secure key

    const int numRuns = 1;
    double totalTime = 0.0;

    for (int i = 0; i < numRuns; ++i){
        // start timer
        auto start = std::chrono::high_resolution_clock::now();

        // allocate host memory for matrices A and B
        unsigned int size_A = WA * HA;
        unsigned int mem_size_A = sizeof(uint) * size_A;
        uint* h_A = (uint*) malloc(mem_size_A);
        unsigned int size_B = WB * HB;
        unsigned int mem_size_B = sizeof(uint) * size_B;
        uint* h_B = (uint*) malloc(mem_size_B);

        // initialize host memory
        randomInit(h_A, size_A);
        randomInit(h_B, size_B);

        // @@@ LITE:CPU Encrypt NxN elements
        int TOTAL_DIM = MATRIX_SIZE * BLOCK_SIZE * MATRIX_SIZE * BLOCK_SIZE;
        uint *enc_A = new uint[TOTAL_DIM];
        uint *enc_B = new uint[TOTAL_DIM];
        ltEncryptCPU(enc_A, h_A, e_sched, Nr, TOTAL_DIM); 
        ltEncryptCPU(enc_B, h_B, e_sched, Nr, TOTAL_DIM);
        // @@@ END-LITE:CPU Encrypt NxN elements
        printf("Matrix A\n");
        for(int k = 0; k < HA; k++){
            for(int j = 0; j < WA; j++){
                printf("%u ", h_A[k * WA + j]);
            }
            printf("\n");
        }
        printf("Matrix B\n");
        for(int k = 0; k < HA; k++){
            for(int j = 0; j < WA; j++){
                printf("%u ", h_B[k * WA + j]);
            }
            printf("\n");
        }

        // printf("Encrypted Matrix A\n");
        // for(int k = 0; k < HA; k++){
        //     for(int j = 0; j < WA; j++){
        //         printf("%u ", enc_a[k * WA + j]);
        //     }
        //     printf("\n");
        // }

        // allocate device memory
        uint* d_A;
        (cudaMalloc((void**) &d_A, mem_size_A));
        uint* d_B;
        (cudaMalloc((void**) &d_B, mem_size_B));

        // copy host memory to device
        (cudaMemcpy(d_A, enc_A, mem_size_A,
                                cudaMemcpyHostToDevice) );
        (cudaMemcpy(d_B, enc_B, mem_size_B,
                                cudaMemcpyHostToDevice) );

        // allocate device memory for result
        unsigned int size_C = WC * HC;
        unsigned int mem_size_C = sizeof(uint) * size_C;
        uint* d_C;
        (cudaMalloc((void**) &d_C, mem_size_C));
        
        // setup execution parameters
        dim3 threads;
        threads.x = threads.y = BLOCK_SIZE; // 4
        dim3 grid;
        grid.x = WC / threads.x; // 20 / 4 = 5
        grid.y = HC / threads.y; // 20 / 4 = 5
        threads.z = grid.z = 1;

        // matrix size = 20 x 20 = 400
        // printf("Number of blocks: %d\n", grid.x * grid.y); // 25
        // printf("Number of threads per block: %d\n", threads.x * threads.y); // 16
        // Grid x Threads = 25 x 16 = 400, one thread process one element result matrix

        // execute the kernel
        matrixMulSecure<<< grid, threads >>>(d_C, d_A, d_B, WA, WB, d_enc_sched, d_dec_sched, Nr);

        // allocate mem for the result on host side
        uint* h_C = (uint*) malloc(mem_size_C);

        // copy result from device to host
        (cudaMemcpy(h_C, d_C, mem_size_C,
                                cudaMemcpyDeviceToHost) );

        // stop timer
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // accumulate execution time
        totalTime += duration.count();

        // check if the matrix multiplication is correct
        if (checkResult(h_C, h_A, h_B, WA, WB, WC, HC))
            printf("%d Pass\n",i);
        else
            printf("%d Fail\n",i);

        
        // clean up memory  
        free(h_A);
        free(h_B);
        (cudaFree(d_A));
        (cudaFree(d_B));
        (cudaFree(d_C));
    }


    // calculate average execution time
    double avgTime = totalTime / numRuns;

    // print average execution time in milliseconds
    printf("Average execution time over %d runs: %.3f ms\n", numRuns, avgTime / 1000.0);

    return 0;
}

// Allocates a matrix with random uint entries.
void randomInit(uint* data, int size)
{
    int i;
    for (i = 0; i < size; ++i)
        data[i] = (uint)rand()%100;
}

// Checks if the matrix multiplication is correct
bool checkResult(uint* C, uint* A, uint* B, int wA, int wB, int wC, int hC)
{
    for (int i = 0; i < hC; ++i)
    {
        for (int j = 0; j < wC; ++j)
        {
            uint sum = 0;
            for (int k = 0; k < wA; ++k)
                sum += A[i * wA + k] * B[k * wB + j];
            if (fabs(C[i * wC + j] - sum) > 1e-5)
                return false;
        }
    }
    return true;
}