#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include <chrono>

#include "lite.cu"

__global__ void vectorAdditionUnsecure(uint *result, uint *a, uint *b, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N; i += stride){
        result[i] = a[i] + b[i];
    }
}

void check(uint *a, uint *b, uint *array, int N){
    bool flag = false;
    for(int i = 0; i < N; i++){
        if(array[i] != a[i]+b[i]){
            flag = true;
        }
    }
    if(!flag){
        printf("SUCCESS\n");
        // for(int i = 0; i < N; i++) printf("%u ", a[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%u ", b[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%u ", array[i]); printf("\n");
    }else{
        printf("FAIL\n");
        // for(int i = 0; i < N; i++) printf("%u ", a[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%u ", b[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%u ", array[i]); printf("\n");
    }
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int N = 1000000; // vector length

    uint *a = new uint[N];
    uint *b = new uint[N];
    uint *c = new uint[N];

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    int runs = 100;
    double avg_time[3] = {0,0,0};
    for(int test=0; test < 3; test++){
        for(int i=0;i<runs;i++){

            // initiate
            for(int ii = 0; ii < N; ii++) {
                a[ii] = rand()%100;
                b[ii] = rand()%100;
            }

            auto t1 = high_resolution_clock::now();

            if(test==1){
                ltVectorAddition(c, a, b, N, e_sched, d_sched, Nr); // secure LITE's function
            }else if(test==0){
                uint *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, sizeof(uint)*N); cudaMemcpy(d_a, a, sizeof(uint)*N, cudaMemcpyHostToDevice);
                cudaMalloc(&d_b, sizeof(uint)*N); cudaMemcpy(d_b, b, sizeof(uint)*N, cudaMemcpyHostToDevice);
                cudaMalloc(&d_c, sizeof(uint)*N);
                vectorAdditionUnsecure<<<1024, 128>>>(d_c, d_a, d_b, N);      // unsecure
                cudaMemcpy(c, d_c, sizeof(uint)*N, cudaMemcpyDeviceToHost);
                cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
            }else{
                for(int i=0;i<N;i++){
                    c[i] = a[i]+b[i];
                }
            }

            auto t2 = high_resolution_clock::now();
            
            duration<double, std::milli> ms_double = t2 - t1;
            avg_time[test] += ms_double.count();
            printf("%d  ", i); check(a, b, c, N);
            cudaDeviceReset();
        }
        avg_time[test]/=runs;
    }
    cout << "GPU unsecure: " << avg_time[0]<< " ms\n"
         << "GPU secure  : " << avg_time[1] << " ms\n"
         << "CPU         : " << avg_time[2] << " ms\n";
}

// unsecure: 66.9532 ms
// secure  : 157.109 ms
