#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>   // C++ version
#include <stdio.h>
#include <iostream>
__global__ void dotProduct(double* da, double* db,double * partialSum, int N) {
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ double cache[256];
	double temp = 0.0f;
	int localId = threadIdx.x;
	if (threadId < N) {
		temp = da[threadId] * db[threadId];
	}
	cache[localId] = temp;
	__syncthreads();
	int strides = blockDim.x / 2;
	while (strides > 0) {
		if (localId < strides) {
			cache[localId] += cache[localId + strides];
		}
		__syncthreads();
		strides /= 2;

	}
	if (localId == 0) {
		partialSum[blockIdx.x] += cache[0];
	}

}
int main(){
	int N = 100000000;
	size_t size = N * sizeof(double);
	double* ha, * hb, * partialSum;
	double* da, * db, * dpartialSum;
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	ha = (double*)malloc(size);
	hb = (double*)malloc(size);
	partialSum = (double*)malloc(numBlocks * sizeof(double));
	
	for (int i = 0; i < N; i++)
	{
		ha[i] = 0.1f * i;
		hb[i] = 0.2f * i;
	}
	
	cudaMalloc((void**) & da, size);
	cudaMalloc((void**) & db, size);
	cudaMalloc((void**) & dpartialSum, numBlocks * sizeof(double));

	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dpartialSum, partialSum, numBlocks * sizeof(double), cudaMemcpyHostToDevice);

	dotProduct << <numBlocks, blockSize >> > (da, db, dpartialSum, N);
	cudaDeviceSynchronize();
	cudaMemcpy(partialSum, dpartialSum, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);
	
	double dotProductResult = 0.0f;
	for (int i = 0; i < numBlocks; i++)
	{
		dotProductResult += partialSum[i];
	}
	std::cout << "Dot Product Result:" << dotProductResult << std::endl;
	
	free(ha);
	free(hb);
	free(partialSum);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dpartialSum);
	return 0;
}