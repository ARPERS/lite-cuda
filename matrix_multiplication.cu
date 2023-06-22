#include <iostream>
#include <cuda.h>

#include "lite.cu"

#define TILE_SIZE 4

using namespace std;

void check(uint target, uint *a, int N){
    bool flag = false;
    for(int i = 0; i < N * N; ++i){
        if(a[i] != target){
            flag = true;
        }
    }
    if(!flag){
        cout << "SUCCESS" << endl;
    }else{
        cout << "FAIL" << endl;
    }
}

int main(){
    int N = 8;  // Matrix size
    int size = N * N * sizeof(uint);

    // Allocate host 
    uint *h_A = new uint[N * N];
    uint *h_B = new uint[N * N];
    uint *h_C = new uint[N * N];

    // initialize
    for (int i = 0; i < N * N; ++i){
        h_A[i] = 2;
        h_B[i] = 4;
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

    ltMatrixMultiplication(h_C, h_A, h_B, N, e_sched, d_sched, Nr);

    for(int i=0;i<N*N;i++) cout << h_C[i] << " " ;
    check(2*4*N, h_C, N);    
}
