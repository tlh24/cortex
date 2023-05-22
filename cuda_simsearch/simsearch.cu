#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include "simsearch.h"

// this (w/ reduce) takes 4.2ms on 3080 laptop = slow.
__global__ void compute_distances(float* db, float* query, float* distances) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < DB_SIZE) {
        float distance = 0;
        for (int i = 0; i < DIM; i++) {
            float diff = db[idx * DIM + i] - query[i];
            distance += diff * diff;
        }
        distances[idx] = distance;
    }
}

__global__ void compute_distances2(unsigned char* db, unsigned char* query, float* dist)
{
	__shared__ float sacc[BLOCK_STRIDE][32];
	//__shared__ unsigned char qq[DIM]; // this makes no difference. 
	// hidden by the HW cache. 
	
	int x = threadIdx.x;
	int y = threadIdx.y;
	int by = blockIdx.x * BLOCK_STRIDE + threadIdx.y;

	float acc = 0.0;
	for( int i = 0; i < WARPSTEPS; i++ ){
		int j = i*32 + x;
		if(j < DIMSHORT){
			float diff = (float)(db[by*DIM + j] - query[j]);
			acc += diff * diff;
		}
	}
	sacc[y][x] = acc;
	__syncthreads();
	if(x < 16) sacc[y][x] += sacc[y][x + 16];
	if(x < 8 ) sacc[y][x] += sacc[y][x + 8 ];
	if(x < 4 ) sacc[y][x] += sacc[y][x + 4 ];
	if(x < 2 ) sacc[y][x] += sacc[y][x + 2 ];
	if(x < 1 ) sacc[y][x] += sacc[y][x + 1 ];

	dist[by] = sacc[y][x];
}

__device__ void warpReduce(volatile float *minDist, 
									volatile int *minIndx, unsigned int tid) {
	// all threads in a warp are synchronous
	// so you don't need to call syncthreads
	// and don't need if(tid < stride)
	for( int stride = 32; stride > 0; stride /= 2){
		if(minDist[tid] > minDist[tid + stride]){
			minDist[tid] = minDist[tid + stride]; 
			minIndx[tid] = minIndx[tid + stride]; 
		}
	}
}  

__global__ void findMinOfArray(float *dists, float *outDist, int* outIndx)
{
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; /* unique id for each thread in the block*/

	unsigned int thread_id = threadIdx.x; /* thread index in the block*/

	__shared__ float minDist[BLOCK_SIZE];
	__shared__ int minIndx[BLOCK_SIZE];

	// load local data. 
	if( row < DB_SIZE ){
		minDist[thread_id] = dists[row];
		minIndx[thread_id] = row; 
	}
	__syncthreads();

	for(unsigned int stride = (blockDim.x/2); stride > 32 ; stride /=2){
		__syncthreads();
		if(thread_id < stride){
			if(minDist[thread_id] > minDist[thread_id + stride]){ 
				minDist[thread_id] = minDist[thread_id + stride]; 
				minIndx[thread_id] = minIndx[thread_id + stride]; 
			}
		}
	}

	if(thread_id < 32){
		warpReduce(minDist, minIndx, thread_id);
	}

	if(thread_id == 0){
		outDist[blockIdx.x] = minDist[0];
		outIndx[blockIdx.x] = minIndx[0];
	}
}

imgdb* simdb_allocate(int num)
{
	if(num != DB_SIZE){
		printf("simdb_allocate: asked for %d, compiled with %d\n",
			num, DB_SIZE);
		return 0;
	}
	imgdb* sdb = new imgdb;

	cudaMalloc(&(sdb->db), DB_SIZE * DIM * sizeof(unsigned char));
	cudaMalloc(&(sdb->query), DIM * sizeof(unsigned char));
	cudaMalloc(&(sdb->distances), DB_SIZE * sizeof(float));
	cudaMalloc(&(sdb->outDist), NUM_BLOCKS * sizeof(float));
	cudaMalloc(&(sdb->outIndx), NUM_BLOCKS * sizeof(int));

	sdb->h_outDist = (float*)malloc(NUM_BLOCKS*sizeof(float));
	sdb->h_outIndx = (int*)malloc(NUM_BLOCKS*sizeof(int));

	return sdb;
}

void simdb_free(imgdb* sdb)
{
	cudaFree(sdb->db);
	cudaFree(sdb->query);
	cudaFree(sdb->distances);
	cudaFree(sdb->outDist);
	cudaFree(sdb->outIndx);

	free(sdb->h_outDist);
	free(sdb->h_outIndx);
	free(sdb);
}

void simdb_set(imgdb* sdb, int i, unsigned char* row)
{
	if( i>=0 && i < DB_SIZE)
		cudaMemcpy(sdb->db + i*DIM, row, DIMSHORT,
				  cudaMemcpyHostToDevice);
}

void simdb_query(imgdb* sdb, unsigned char* query,
				float* minDist, int* minIndx)
{
	cudaMemcpy(sdb->query, query, DIMSHORT,
				  cudaMemcpyHostToDevice);

	dim3 dimBlock(32, BLOCK_STRIDE, 1); // x, y, z

	compute_distances2<<<DB_SIZE/BLOCK_STRIDE, dimBlock>>>
			(sdb->db, sdb->query, sdb->distances);

	findMinOfArray<<<NUM_BLOCKS, BLOCK_SIZE>>>
			(sdb->distances, sdb->outDist, sdb->outIndx);

	cudaMemcpy(sdb->h_outDist, sdb->outDist, sizeof(float)*NUM_BLOCKS, cudaMemcpyDeviceToHost);
	cudaMemcpy(sdb->h_outIndx, sdb->outIndx, sizeof(int)*NUM_BLOCKS, cudaMemcpyDeviceToHost);

	float d = sdb->h_outDist[0];
	float n = sdb->h_outIndx[0];
	for(int i=1; i<NUM_BLOCKS; i++){
		if(sdb->h_outDist[i] < d){
			d = sdb->h_outDist[i];
			n = sdb->h_outIndx[i];
		}
	}
	*minDist = d;
	*minIndx = n;
}
