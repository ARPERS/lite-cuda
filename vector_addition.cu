#include <stdio.h>

__global__ void vectorAddition(float *result, float *a, float *b, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}

void check(float target, float *array, int N){
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
    size_t size = N * sizeof(float);

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    float *d_a, *d_b, *d_c;
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

  
    int threadsPerBlock;
    int numberOfBlocks;

    threadsPerBlock = 16;
    numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddition<<<numberOfBlocks, threadsPerBlock>>>(d_c, d_a, d_b, N);
    
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    check(5, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
