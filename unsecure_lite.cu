// USED FOR BENCHMARKING
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "AES/AES_encrypt_cpu.cpp"
#include "AES/AES_encrypt_gpu.cu"
#include "AES/AES_decrypt_cpu.cpp"
#include "AES/AES_decrypt_gpu.cu"
#include "AES/AES.cu"

using namespace std;

#include "lite_utils.cu"
#include "lite_encdec.cu"  
#include "unsecure_lite_vector.cu"
#include "unsecure_lite_matrix.cu"

