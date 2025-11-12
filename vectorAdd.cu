
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>   // C++ version
#include <stdio.h>
#include <iostream>
using namespace std;


__global__ void addVectors(const float* a, const float* b, float* c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		c[i] = a[i] + b[i];
	}
}

int main()
{
	int N = 100;
	size_t size = N * sizeof(float);
	float* ha, * hb, * hc;
	ha = (float*)malloc(size);
	hb = (float*)malloc(size);
	hc = (float*)malloc(size);
	for (int i = 0; i < N; i++) {
		ha[i] = i * 1.0f;
		hb[i] = i * 2.0f;
	}
	float* da, * db, * dc;
	cudaMalloc(&da, size);
	cudaMalloc(&db, size);
	cudaMalloc(&dc, size);
	cudaMemcpy(da, ha, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	addVectors << <numBlocks, blockSize >> > (da, db, dc, N);
	cudaDeviceSynchronize();
	cudaMemcpy(hc, dc, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		std::cout << hc[i] << std::endl;
	}
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	free(ha);
	free(hb);
	free(hc);
	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
