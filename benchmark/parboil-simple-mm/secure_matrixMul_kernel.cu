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
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#ifdef __MCUDA__
#include <mcuda.h>
#endif

#include "matrixMul.h"
#include "../../lib/lite.h"

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMulSecure( uint* C, uint* A, uint* B, int wA, int wB, uint *d_enc_sched, uint *d_dec_sched, int Nr)
{
    // fflush(stdout);
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    __shared__ uint Cs[BLOCK_SIZE][BLOCK_SIZE]; // LITE HERE    
    Cs[tx][ty] = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    int a, b;
    for (a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ uint As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ uint Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // LITE HERE DECRYPT
        AES_decrypt_gpu(As[ty], As[ty], d_dec_sched, Nr); 
        AES_decrypt_gpu(Bs[ty], Bs[ty], d_dec_sched, Nr); 
        // __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        int k;
        for (k = 0; k < BLOCK_SIZE; ++k){
            Cs[ty][tx] += As[ty][k] * Bs[k][tx];
            // if(blockIdx.x == 0 && blockIdx.y == 0){ // for debugging
            //     printf("(a=%d, b=%d) (Tx %d, Ty %d) (Blx %d, Bly %d), k %d, As[ty][k] %d, Bs[k][tx] %d\n", a,b, tx, ty, bx, by, k, As[ty][k], Bs[k][tx]);
            // }
        }

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c;
    c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;

    // LITE ENCRYPT HERE SINGLE ELEMENT
    AES_encrypt_gpu(Cs[ty], Cs[ty], d_enc_sched, Nr); 

    C[c + wB * ty + tx] = Cs[ty][tx];
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
