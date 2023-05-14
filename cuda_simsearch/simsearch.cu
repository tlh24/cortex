#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

#define DIM 900
#define DB_SIZE 100000
#define BLOCK_SIZE 256

using namespace std;

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

__global__ void find_global_best_match(int* best_match, float* min_distance, int* global_best_match, float* global_min_distance) {
    __shared__ int indices[256];
    __shared__ float dists[256];

    int idx = threadIdx.x;

    // Each thread loads its distance and index into shared memory
    indices[threadIdx.x] = best_match[idx];
    dists[threadIdx.x] = min_distance[idx];

    __syncthreads();

    // Reduction to find the minimum distance and corresponding index
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && dists[threadIdx.x] > dists[threadIdx.x + s]) {
            dists[threadIdx.x] = dists[threadIdx.x + s];
            indices[threadIdx.x] = indices[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the result to global memory
    if (threadIdx.x == 0) {
        *global_best_match = indices[0];
        *global_min_distance = dists[0];
    }
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

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

int main() {
	float* db;
	float* query;
	float* distances;
	float* outDist;
	int*   outIndx;
	
	int num_blocks = std::ceil(static_cast<float>(DB_SIZE) / 256.0);
	printf("num blocks: %d\n", num_blocks); 

	// Allocate memory for the database, query vector, distances, and best match index
	cudaMalloc(&db, DB_SIZE * DIM * sizeof(float));
	cudaMalloc(&query, DIM * sizeof(float));
	cudaMalloc(&distances, DB_SIZE * sizeof(float));
	cudaMalloc(&outDist, num_blocks * sizeof(int));
	cudaMalloc(&outIndx, num_blocks * sizeof(float));

	// Initialize the database and query vector
	// Create a random number generator
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(time(NULL)));

	// Fill the database with normally distributed random numbers
	curandGenerateNormal(gen, db, DB_SIZE * DIM, 0.0f, 1.0f);

	// Fill the query vector with normally distributed random numbers
	curandGenerateNormal(gen, query, DIM, 0.0f, 1.0f);

	// Destroy the generator
	curandDestroyGenerator(gen);
	
	// Choose a random index
	std::random_device rd;
	std::mt19937 generator(rd());

	// Now generate random numbers with generator
	std::uniform_int_distribution<int> distribution(1, DB_SIZE);
	int random_indx = distribution(generator);

	// Copy the query vector to the random index in the database
	cudaMemcpy(db + random_indx * DIM, query, DIM * sizeof(float), cudaMemcpyDeviceToDevice);
	
	float* h_outDist = (float*)malloc(num_blocks*sizeof(float));
	int* h_outIndx = (int*)malloc(num_blocks*sizeof(int));
	float minDist; 
	int minIndx; 
	
	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

	for(int u = 0; u < 10; u++){
		// Compute the L2 distances
		compute_distances<<<num_blocks, 256>>>(db, query, distances);

		findMinOfArray<<<num_blocks, 256>>>(distances, outDist, outIndx);
		
		cudaMemcpy(h_outDist, outDist, sizeof(float)*num_blocks, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_outIndx, outIndx, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
		
		minDist = h_outDist[0]; 
		minIndx = h_outIndx[0]; 
		for(int i=1; i<num_blocks; i++){
			if(h_outDist[i] < minDist){
				minDist = h_outDist[i]; 
				minIndx = h_outIndx[i]; 
			}
		}
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
	
	timespec duration = diff(time1, time2); 
	cout<<duration.tv_nsec / 1e10 <<endl;
	printf("Best match: %d, should be %d; Minimum distance: %f\n", 
			 minIndx, random_indx, minDist);

	// Free the memory
	cudaFree(db);
	cudaFree(query);
	cudaFree(distances);
	cudaFree(outDist);
	cudaFree(outIndx);
	free(h_outDist); 
	free(h_outIndx); 

	return 0;
}
