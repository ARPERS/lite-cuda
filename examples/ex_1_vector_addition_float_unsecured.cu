/*
Example for float vector addition unsecure way
This code include the benchmarking code. See "benchmark run" in main() function.
*/
#include <stdio.h>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <chrono>
#include "unsecure_lite.cu"

void check(float *a, float *b, float *array, int N){
    bool flag = false;
    for(int i = 0; i < N; i++){
        if(array[i] != a[i]+b[i]){
            flag = true;
        }
    }
    if(!flag){
        printf("SUCCESS\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", a[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", b[i]); printf("\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", array[i]); printf("\n");
    }else{
        printf("FAIL\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", a[i]); printf("\n"); printf("\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", b[i]); printf("\n"); printf("\n");
        // for(int i = 0; i < N; i++) printf("%.3f ", array[i]); printf("\n");
    }
}

int main() {
    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    int N = 500; // vector length

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];


   vector<double> times;
    for(int i = 0; i < 50; i++){ // benchmark run
        // initiate
        for(int i = 0; i < N; i++) {
            a[i] = rand()%100 / 10.0;
            b[i] = rand()%100 / 10.0;
        }
        
        auto t1 = high_resolution_clock::now();
        
        ltVecAdd(c, a, b, N);
        
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        times.push_back(ms_double.count());
        
        check(a, b, c, N);
        cudaDeviceReset();
    }
    // average time
    double sum = 0;
    for(int i = 0; i < times.size(); i++){
        sum += times[i];
    }
    printf("Average time: %.3f ms\n", sum/times.size());
}
