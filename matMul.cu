#include<stdio.h>
#include<conio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <iostream>
#include <iomanip>

__global__ void matrixMul(const double * da,const double *db,double *dc,int N){
    int row=blockIdx.y*blockDim.y + threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    double value=0.0;
    if(row<N && col<N){
        for(int k=0;k<N;k++){
            value+=da[row*N+k]*db[k*N+col];
        }
    }
    dc[row*N+col]=value;
}
int main(){
    int N=1000;
    double *ha,*hb,*hc;
    size_t size= N*N*sizeof(double);
    ha=(double *)malloc(size);
    hb=(double*) malloc(size);
    hc=(double*)malloc(size);
    for(int i=0;i<N*N;i++){
        ha[i]=0.1;
        hb[i]=0.2;
    }
    double *da,*db,*dc;
    cudaMalloc((void **)&da,size);
    cudaMalloc((void **)&db,size);
    cudaMalloc((void**)&dc,size);
    cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);
    dim3 threadsperblock=dim3(16,16);
    dim3 numBlocks=dim3((N+15)/16,(N+15)/16);
    matrixMul<<<numBlocks,threadsperblock>>>(da,db,dc,N);
    cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);
   std::cout << std::fixed << std::setprecision(17);
std::cout << "C[0] = " << hc[0] << std::endl;

    free(ha);free(hb);free(hc);
    cudaFree(da);cudaFree(da);cudaFree(dc);
    return 0;
}