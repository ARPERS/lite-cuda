#include <stdio.h>
#include <iostream>
#include <typeinfo>

#include "lite.cu"

void check(float target, float *array, int N){
    bool flag = false;
    for(int i = 0; i < N; i++){
        if(array[i] != target){
            flag = true;
        }
    }
    if(!flag){
        printf("SUCCESS\n");
        for(int i = 0; i < N; i++) printf("%.3f ", array[i]); printf("\n");
    }else{
        printf("FAIL\n");
        for(int i = 0; i < N; i++) printf("%.3f ", array[i]); printf("\n");
    }
}

int main() {
    int N = 12; // vector length

    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    uchar key[] = { 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00 };
    uint keySize = 16;
    int Nr=10;
    uint e_sched[4*(MAXNR + 1)];
    uint d_sched[4*(MAXNR + 1)];
    makeKey(key, keySize << 3, DIR_BOTH, e_sched, d_sched, Nr);

    // initiate
    for(int i = 0; i < N; i++) {
        a[i] = 3.1;
        b[i] = 2.5;
    }

    ltVectorAddition(c, a, b, N, e_sched, d_sched, Nr);
    
    check(5.6, c, N);
}
