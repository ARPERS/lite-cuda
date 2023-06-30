#include <iostream>
#include <chrono>
#include <cuda.h>

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
        return;
    }else{
        cout << "FAIL: Result Incorrect!" << endl;
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
    
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    int N = 64;  // Matrix size
    
    double res_time[2];
    double avg_time = 0; int runs=50;
    for(int tt=0;tt<2;tt++){
        for(int ii=0;ii<runs;ii++){     
            auto t1 = high_resolution_clock::now();
            // Allocate host 
            uint *h_A = new uint[N * N];
            uint *h_B = new uint[N * N];
            uint *h_C = new uint[N * N];
        
            uint e_sched[4*(MAXNR + 1)];
            uint d_sched[4*(MAXNR + 1)];
            
            int Nr=10;

            // initialize
            for (int i = 0; i < N * N; ++i){
                h_A[i] = rand()%5+1;
                h_B[i] = rand()%10;
            }

            if(tt==1){
                uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00 };
                uint keySize = 16;
                makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);
            }
            
            ltMatrixMultiplication(h_C, h_A, h_B, N, e_sched, d_sched, Nr, tt);
            auto t2 = high_resolution_clock::now();

            duration<double, std::milli> ms_double = t2 - t1;
            avg_time += ms_double.count();

            check(h_A, h_B, h_C, N);    
        }
        res_time[tt] = avg_time/runs;
    }
    cout << "unsecure: " << res_time[0] << " ms\nsecure  : " << res_time[1] << " ms\n";

}

// average time from 50 runs 64x64 matrix
// unsecure 2.67490 ms
// secure   3.11648 ms
