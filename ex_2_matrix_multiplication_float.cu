#include <iostream>
#include <cuda.h>

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
            if(abs(c[i*N+j]-res[i*N+j]) > 0.00001){
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
    //         printf("%.4f ", c[i*N+j]);
    // printf("\nRES: ");
    // for(int i = 0; i < N; ++i)
    //     for(int j = 0; j < N; ++j)
    //         printf("%.4f ", res[i*N+j]);
    // printf("\n");
}

int main(){
    int N = 4;  // Matrix size

    // Allocate host 
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    // initialize
    for (int i = 0; i < N * N; ++i){
        h_A[i] = rand()%5+1 + (rand()%2+2)/10.0;
        h_B[i] = rand()%10  + (rand()%2+2)/10.0;
    }

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    liteMatMultiplication(h_C, h_A, h_B, N, e_sched, d_sched, Nr, true); // change to false for unsecure running (for benchmarking)

    check(h_A, h_B, h_C, N);    
}
