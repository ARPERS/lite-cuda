#include <stdio.h>
#include <typeinfo>

__global__ void vectorAddition(uint *result, uint *a, uint *b, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}

void check(uint target, uint *array, int N){
    bool flag = false;
    for(int i = 0; i < N; i++){
        if(array[i] != target){
            flag = true;
        }
    }
    if(!flag){
        printf("SUCCESS\n");
    }else{
        printf("FAIL\n");
    }
}

int main() {
    int N = 20; // vector length
    size_t size = N * sizeof(uint);

    uint *a = new uint[N];
    uint *b = new uint[N];
    uint *c = new uint[N];

    uint *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // initiate
    for(int i = 0; i < N; ++i) {
        a[i] = 2.0;
        b[i] = 3.0;
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  
    vectorAddition<<<1, N>>>(d_c, d_a, d_b, N);
    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    check(5, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
