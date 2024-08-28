// USED FOR BENCHMARKING
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <cuda_runtime.h>

#include "../lib/AES/AES_encrypt_cpu.cpp"
#include "../lib/AES/AES_encrypt_gpu.cu"
#include "../lib/AES/AES_decrypt_cpu.cpp"
#include "../lib/AES/AES_decrypt_gpu.cu"
#include "../lib/AES/AES.cu"

using namespace std;

#include "../lib/lite.cu"
#include "../lib/lite.h"
#include "unsecure_lite_vector.cu"
