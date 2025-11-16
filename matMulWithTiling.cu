#include<cuda_runtime.h>
#include<stdio.h>
#include<device_launch_parameters.h>
#define TILE_SIZE 16
#include<iostream>
__global__ void optimizedMatMul(const float*A,const float*B, float*C,int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    float value=0.0f;
    int row= blockIdx.y*TILE_SIZE+threadIdx.y;
    int col=blockIdx.x*TILE_SIZE+threadIdx.x;
    for(int t=0;t<(N+TILE_SIZE-1)/TILE_SIZE;t++)
    {
        if(row<N && threadIdx.x+(t*TILE_SIZE)<N ){
            As[threadIdx.y][threadIdx.x]=A[row*N+t*TILE_SIZE+threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x]=0.0;
        }
        if(col<N && t*TILE_SIZE+threadIdx.y<N){
            Bs[threadIdx.y][threadIdx.x]=B[(t*TILE_SIZE+threadIdx.y)*N+col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x]=0.0;
        }
        __syncthreads();
        for(int k=0;k<TILE_SIZE;k++){
            value+=As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
        __syncthreads();
    
    }
    C[row*N+col]=value;
}
int main(){
    int N=2048;
    size_t size=sizeof(float)*N*N;
    float *ha, *hb,*hc;
    ha=(float*)malloc(size);
    hb=(float*)malloc(size);
    hc=(float*)malloc(size);
    for(int i=0;i<N*N;i++){
        ha[i]=0.1f;
        hb[i]=0.2f;
    }
    float *da,*db,*dc;
    cudaMalloc((void**)&da,size);
    cudaMalloc((void**)&db,size);
    cudaMalloc((void**)&dc,size);

    cudaMemcpy(da,ha,size,cudaMemcpyHostToDevice);
    cudaMemcpy(db,hb,size,cudaMemcpyHostToDevice);

    dim3 thread(TILE_SIZE,TILE_SIZE);
    dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE,(N+TILE_SIZE-1)/TILE_SIZE);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    optimizedMatMul<<<blocks,thread>>>(da,db,dc,N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Time (ms): " << milliseconds << std::endl;
    cudaMemcpy(hc,dc,size,cudaMemcpyDeviceToHost);
    std::cout<< "C[0]" <<hc[0]<< std::endl;
    cudaFree(da);cudaFree(db);cudaFree(dc);
    free(ha);free(hb);free(hc);

}