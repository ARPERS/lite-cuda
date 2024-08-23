#ifndef _LITE_UTILS_H_
#define _LITE_UTILS_H_

#include "lite.h"

// Debugging macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Device functions
__device__ unsigned int* floatToUint(float *input);
__device__ float* uintToFloat(unsigned int *input);

// Host functions
void floatToUintCPU(uint *dest, const float *source, int N);
void uintToFloatCPU(float *dest, const uint *source, int N);
int padArray(uint* arr, int N);
void removePadArray(uint* arr, int N, int paddingSize);
int padMatrix(uint* matrix, int& width, int& height);
void removePadMatrix(int* matrix, int& width, int& height, int paddingSize);

#endif // _LITE_UTILS_H_
