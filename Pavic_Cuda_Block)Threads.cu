
// Pavic  - CUDA -  Add Block & Threads

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<iostream>
#include<random>
#include <stdio.h>


__global__ void whoami(void) {
	int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x + gridDim.y;
	int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
	int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y;
	int ID = block_offset + thread_offset;

	printf("%04d | block(%d %d %d))= %3d !thread(% d % d % d) = %3d\n", ID, blockIdx.x, blockIdx.y, blockIdx.z, thread_offset);

}

//populate vectors with random ints
void random_ints(int* a, int N) {
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 1000;
	}
}
//  cuda = ADD() BLOCK
__global__ void add_blocks(int* a, int* b, int* c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

//  cuda = ADD() THREADS
__global__ void add_threads(int* a, int* b, int* c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}
// CUDA Index (Thread and Blocks)
__global__ void add(int* a, int* b, int* c, int n ) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index<n)
	c[index] = a[index] + b[index];
}


//#define N 51200000             // Parallel Problem !!


// With More Parallel computing power, Solve bigger Problems!! 
#define N (2048*2048) // Bigger Problem
#define THREADS_PER_BLOCK 512


int main(void) {
	int* a, * b, * c;	// host copies of a, b, c
	int* d_a, * d_b, * d_c;	// device copies of a, b, c
	int size = N * sizeof(int); // More Memory!!

	// Alloc space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Alloc space for host copies of a, b, c and setup input values

	a = (int*)malloc(size); 
	
	random_ints(a, N);
	b = (int*)malloc(size);
	random_ints(b, N);

	c = (int*)malloc(size);

		// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N blocks
	//add_blocks <<<N, 1 >> > (d_a, d_b, d_c); // Parallel - Blocks
	//add_threads <<<1, N >> > (d_a, d_b, d_c); // PArallel - Threads
	
	// Launch add() kernel on GPU with threads & Blocks
	// Now the Function add<<<Blocks, threads >>>(d_a, d_b, d_c);
	// N (2048*2048)
	// THREADS_PER_BLOCK 512
	add <<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c, N);

	whoami << < N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> ();
	cudaDeviceSynchronize();


	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	printf(" Sixe of C= %d", size);
/*
	for (int i = 0; i < N; i++) {
		printf(c[i]);
		sizeof(c)

	}
	*/
	

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	return 0;
}