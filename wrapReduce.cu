#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>


__inline__ __device__ float wrapper(float val)
{
    unsigned int mask=0xffffffffu;
    for(int offset=16;offset>0;offset/=2)
    {
        val+=__shfl_down_sync(mask,val,offset);
    }
    return val;
}
__global__ void wrapReduce(float * in, float *out,int N)
{
    extern __shared__ float sdata[];
    int gloabalIdx=blockDim.x*blockIdx.x+threadIdx.x;
    int lane=gloabalIdx%32;
    int wrapIdx=threadIdx.x>>5;
    float val=in[gloabalIdx];
    float wrapVal=wrapper(val);
    if(lane==0){
        sdata[wrapIdx]=wrapVal;
    }
    __syncthreads();
    if(wrapIdx==0)
    {
        int numWraps=(blockDim.x+31)/32;
        float blockSum=0.0f;
        if(threadIdx.x<numWraps){
            blockSum=sdata[threadIdx.x];
        }
        else{
            blockSum=0.0f;
        }
        float wrappedSum=wrapper(blockSum);
        if(threadIdx.x==0){
        out[blockIdx.x]=wrappedSum;
    }
    }
}
int main(){
    int N=1<<24;
    size_t size=N*sizeof(float);
    float *input,*output;
    float *d_input,*host_output;

    input=(float*)malloc(size);
    for(int i=0;i<N;i++){
        input[i]=0.01f;
    }
    int blockSize=256;
    int numBlocks=(N+blockSize-1)/blockSize;
    cudaMalloc((void**)&output,numBlocks*sizeof(float));
    cudaMalloc((void**)& d_input,size);
    host_output=(float*)malloc(numBlocks*sizeof(float));
    cudaMemcpy(d_input,input,size,cudaMemcpyHostToDevice);
    size_t shmemSize=((blockSize+31)/32)*sizeof(float);
    wrapReduce<<<numBlocks,blockSize,shmemSize>>>(d_input,output,N);
    cudaDeviceSynchronize();
    cudaMemcpy(host_output, output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Partial block sums:\n";
    double finalSum = 0.0;

    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Block " << i << ": " << host_output[i] << "\n";
        finalSum += host_output[i];
    }

    std::cout << "\nTotal sum = " << finalSum << "\n";
    std::cout << "Expected sum = " << (N * 0.01f) << "\n";

    return 0;

}