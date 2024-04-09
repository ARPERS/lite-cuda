/*
Example for Float Matrix Multiplication using LITE
*/

#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>

#include <cstdlib> // for atoi
#include <cstring> // for strcmp

#include "lite.cu"

using namespace std;

void check(float *a, float *b, float *res, int N){
    bool flag = false;
    float *c = new float[N*N];
    for(int i = 0; i < N; ++i) for(int j = 0; j < N; ++j) c[i*N+j] = 0;

    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            for(int k = 0; k < N; ++k){
                int row_a = i;
                int col_b = j;
                int cr_ab = k;
                c[row_a*N + col_b] += a[row_a*N + cr_ab]*b[cr_ab*N + col_b];
            }
    
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            if(abs(c[i*N+j]-res[i*N+j]) > 0.001){
                flag = true;
                printf("ERROR: %.4f != %.4f diff %.8f\n", c[i*N+j], res[i*N+j], abs(c[i*N+j]-res[i*N+j]));
            }
    if(!flag){
        cout << "SUCCESS" << endl;
    }else{
        cout << "FAIL" << endl;
    }
    // printf("ANS: ");
    // for(int i = 0; i < N; ++i){
    //     for(int j = 0; j < N; ++j)
    //         printf("%.4f ", c[i*N+j]);
    //     printf("\n");
    // }
    // printf("\nRES: ");
    // for(int i = 0; i < N; ++i){
    //     for(int j = 0; j < N; ++j)
    //         printf("%.4f ", res[i*N+j]);
    //     printf("\n");
    // }
    // printf("\n");
}

int main(int argc, char* argv[]){

    using chrono::high_resolution_clock;
    using chrono::duration;
    using chrono::milliseconds;

    // Check if the correct number of arguments are provided
    if (argc != 3) {
        std::cout << "Please provide two arguments: N and is_secure." << std::endl;
        return 1;
    }
    int N = atoi(argv[1]);
    bool is_secure = strcmp(argv[2], "true") == 0;

    // Allocate host 
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    vector<double> times;
    for(int i = 0; i < 50 + 1; i++){ // benchmark run    
        
        // initialize
        for (int i = 0; i < N * N; ++i){
            h_A[i] = rand()%5+1 + (rand()%2+2)/10.0;
            h_B[i] = rand()%10  + (rand()%2+2)/10.0;
        }

        auto t1 = high_resolution_clock::now(); 

        liteMatMultiplication(h_C, h_A, h_B, N, e_sched, d_sched, Nr, is_secure); // change to false for unsecure running (for benchmarking)
        
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        times.push_back(ms_double.count());

        check(h_A, h_B, h_C, N);  

        cout << "Time: " << ms_double.count() << " ms" << endl;
        cudaDeviceReset();
    }
    // average time
    double sum = 0;
    for(int i = 0; i < times.size(); i++){
        if(i == 0) continue;
        sum += times[i];
    }
    printf("Average time: %.3f ms\n", sum/(times.size()-1));  
}
