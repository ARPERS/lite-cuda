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
#include "matrixMul_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void randomInit(uint*, int);
bool checkResult(uint* C, uint* A, uint* B, int wA, int wB, int wC, int hC);

__global__ void matrixMul(uint* C, uint* A, uint* B, int wA, int wB);

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    printf("UNSECURE BENCHMARK\n");
    printf("WA %d | HA %d | WB %d | HB %d\n", WA, HA, WB, HB);
    // set seed for rand()
    srand(2006);

    const int numRuns = 20;
    double totalTime = 0.0;

    // setup execution parameters
    dim3 threads;
    threads.x = threads.y = BLOCK_SIZE;
    dim3 grid;
    grid.x = WC / threads.x;
    grid.y = HC / threads.y; 
    threads.z = grid.z = 1;

    printf("Number of blocks: %d\n", grid.x * grid.y); // 25
    printf("Number of threads per block: %d\n", threads.x * threads.y); // 16
    // Grid x Threads = 25 x 16 = 400, one thread process one element result C matrix


    for (int i = 0; i < numRuns; ++i)
        {

        // allocate host memory for matrices A and B
        unsigned int size_A = WA * HA;
        unsigned int mem_size_A = sizeof(uint) * size_A;
        uint* h_A = (uint*) malloc(mem_size_A);
        unsigned int size_B = WB * HB;
        unsigned int mem_size_B = sizeof(uint) * size_B;
        uint* h_B = (uint*) malloc(mem_size_B);

        // allocate device memory
        uint* d_A;
        (cudaMalloc((void**) &d_A, mem_size_A));
        uint* d_B;
        (cudaMalloc((void**) &d_B, mem_size_B));

        // allocate device memory for result
        unsigned int size_C = WC * HC;
        unsigned int mem_size_C = sizeof(uint) * size_C;
        uint* d_C;
        (cudaMalloc((void**) &d_C, mem_size_C));

        // allocate mem for the result on host side
        uint* h_C = (uint*) malloc(mem_size_C);

        // initialize host memory
        randomInit(h_A, size_A);
        randomInit(h_B, size_B);

        // copy host memory to device
        (cudaMemcpy(d_A, h_A, mem_size_A,
                                cudaMemcpyHostToDevice) );
        (cudaMemcpy(d_B, h_B, mem_size_B,
                                cudaMemcpyHostToDevice) );

        // start timer
        auto start = std::chrono::high_resolution_clock::now();
        // execute the kernel
        matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
        // stop timer
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // copy result from device to host
        (cudaMemcpy(h_C, d_C, mem_size_C,
                                cudaMemcpyDeviceToHost) );

        // accumulate execution time
        if(i>9)
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
    double avgTime = totalTime / (numRuns);

    // print average execution time in milliseconds
    printf("Average execution time over %d runs: %.6f ms\n", numRuns, avgTime / 1000.0);
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
            if (C[i * wC + j] != sum)
                return false;
        }
    }
    return true;
}