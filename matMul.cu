#include<stdio.h>
#include<conio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <iostream>
#include <iomanip>

__global__ void matrixMul(const float * da,const float *db,float *dc,int N){
    int row=blockIdx.y*blockDim.y + threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    float value=0.0f;
    if(row<N && col<N){
        for(int k=0;k<N;k++){
            value+=da[row*N+k]*db[k*N+col];
        }
    }
    dc[row*N+col]=value;
}
int main(){
    int N=2048;
    float *ha,*hb,*hc;
    size_t size= N*N*sizeof(float);
    ha=(float *)malloc(size);
    hb=(float*) malloc(size);
    hc=(float*)malloc(size);
    for(int i=0;i<N*N;i++){
        ha[i]=0.1f;
        hb[i]=0.2f;
    }
    float *da,*db,*dc;
    cudaMalloc((void **)&da,size);
    cudaMalloc((void **)&db,size);
    cudaMalloc((void**)&dc,size);
    cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);
    dim3 threadsperblock=dim3(16,16);
    dim3 numBlocks=dim3((N+15)/16,(N+15)/16);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMul<<<numBlocks,threadsperblock>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Time (ms): " << milliseconds << std::endl;
    cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);
   std::cout << std::fixed << std::setprecision(17);
std::cout << "C[0] = " << hc[0] << std::endl;

    free(ha);free(hb);free(hc);
    cudaFree(da);cudaFree(da);cudaFree(dc);
    
    return 0;
}