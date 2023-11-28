#include <iostream>
#include <cuda.h>
#include <chrono>
#include <vector>

#include "lite.cu"

#define TILE_SIZE 4

using namespace std;

void check(uint *a, uint *b, uint *res, int N){
    bool flag = false;
    uint *c = new uint[N*N];
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
            if(c[i*N+j]!=res[i*N+j]){
                flag = true;
            }
    if(!flag){
        cout << "SUCCESS" << endl;
    }else{
        cout << "FAIL" << endl;
    }
    // printf("ANS: ");
    // for(int i = 0; i < N; ++i)
    //     for(int j = 0; j < N; ++j)
    //         printf("%u ", c[i*N+j]);
    // printf("\nRES: ");
    // for(int i = 0; i < N; ++i)
    //     for(int j = 0; j < N; ++j)
    //         printf("%u ", res[i*N+j]);
    // printf("\n");
}

int main(){
    
    using chrono::high_resolution_clock;
    using chrono::duration;
    using chrono::milliseconds;

    int N = 200;  // Matrix size
    
    // Allocate host 
    uint *h_A = new uint[N * N];
    uint *h_B = new uint[N * N];
    uint *h_C = new uint[N * N];

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
    for(int i = 0; i < 10; i++){ // benchmark run        
        // initialize
        for (int j = 0; j < N * N; ++j){
            h_A[j] = rand()%5+1;
            h_B[j] = rand()%10;
        }
        
        auto t1 = high_resolution_clock::now();
        liteMatMultiplication(h_C, h_A, h_B, N, e_sched, d_sched, Nr, true);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        times.push_back(ms_double.count());
        
        check(h_A, h_B, h_C, N);   
        cout << "Time: " << ms_double.count() << " ms" << endl;
    }
    // average time
    double sum = 0;
    for(int i = 1; i < times.size(); i++){
        sum += times[i];
    }
    printf("Average time: %.3f ms\n", sum/(times.size()-1));
}