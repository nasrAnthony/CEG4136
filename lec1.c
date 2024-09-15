#include <stdlib.h>
#include <time.h>
#include <stdio.h>


void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for(int idx = 0; idx < N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}


void initData(float *ip, int size){
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}


int main(int argx, char **argv){
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);


    initData(h_A, nElem);
    initData(h_B, nElem);
    initData(h_C, nElem);


    sumArraysOnHost(h_A, h_B, h_C, nElem);

    for(int index = 0; index < nElem; index++){
        printf("%f\n", h_C[index]);
    }

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}




