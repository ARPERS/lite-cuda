#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <chrono>

#include "lite.cu"

__global__ void vectorAddition(uint *result, uint *a, uint *b, int N){
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
        return;
        // for(int i = 0; i < N; i++) printf("%u ", array[i]); printf("\n");
    }else{
        printf("FAIL: Result Incorrect!\n");
        // for(int i = 0; i < N; i++) printf("%u ", array[i]); printf("\n");
    }
}

int main() {

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    double res_time[2];
    int N = 4000; // vector length

    for(int tt=0; tt<2; tt++){

        double avg_time = 0; int runs=200;
        size_t size = N * sizeof(uint);

        for(int ii=0; ii<runs; ii++){
            
            uint *a = new uint[N];
            uint *b = new uint[N];
            uint *c = new uint[N];
            
            // initiate
            for(int i = 0; i < N; ++i) {
                a[i] = rand()%100;
                b[i] = rand()%100;
            }

            if(tt==0){
                // vector addition without LITE
                cudaDeviceSynchronize();
                auto t1 = high_resolution_clock::now();
                uint *d_a, *d_b, *d_c;
                cudaMalloc(&d_a, size);
                cudaMalloc(&d_b, size);
                cudaMalloc(&d_c, size);

                cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
                
                vectorAddition<<<1, N/4>>>(d_c, d_a, d_b, N);
                cudaDeviceSynchronize();

                cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
                
                auto t2 = high_resolution_clock::now();

                duration<double, std::milli> ms_double = t2 - t1;
                avg_time += ms_double.count();
            }else{
                // vector addition with LITE
                cudaDeviceSynchronize();
                auto t1 = high_resolution_clock::now();
                uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00 };
                uint keySize = 16;
                int Nr=10;
                uint e_sched[4*(MAXNR + 1)];
                uint d_sched[4*(MAXNR + 1)];
                makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

                ltVectorAddition(c, a, b, N, e_sched, d_sched, Nr);
                cudaDeviceSynchronize();
                auto t2 = high_resolution_clock::now();

                duration<double, std::milli> ms_double = t2 - t1;
                avg_time += ms_double.count();
            }
            
            check(a, b, c, N);
        }
        res_time[tt]=avg_time/runs;
    }
    cout << "unsecure: " << res_time[0] << " ms\nsecure  : " << res_time[1] << " ms\n";
}