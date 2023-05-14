#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

#define WARPSTEPS 30
#define NUM_BLOCKS 400
#define BLOCK_SIZE 256 // 256 and 512 yield the same bandwidth.
#define BLOCK_STRIDE (BLOCK_SIZE/32)
#define DIM (32 * WARPSTEPS)
#define DB_SIZE (NUM_BLOCKS * BLOCK_SIZE)

using namespace std;

// this w reduce takes 4.2ms on 3080 laptop.
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

__global__ void compute_distances2(float* db, float* query, float* dist)
{
	__shared__ float sacc[BLOCK_STRIDE][32];
	int x = threadIdx.x;
	int y = threadIdx.y;
	int by = blockIdx.x * BLOCK_STRIDE + threadIdx.y;

	float acc = 0.0;
	for( int i = 0; i < WARPSTEPS; i++ ){
		float diff = db[by*DIM + i*32 + x] - query[i*32 + x];
		acc += diff * diff;
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
	
	int num_blocks = NUM_BLOCKS;
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
	dim3 dimBlock(32, BLOCK_STRIDE, 1); // x, y, z
	
	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

	for(int u = 0; u < 10; u++){
		// Compute the L2 distances
		//compute_distances<<<num_blocks, 256>>>(db, query, distances);
		compute_distances2<<<DB_SIZE/BLOCK_STRIDE, dimBlock>>>(db, query, distances);

		findMinOfArray<<<num_blocks, BLOCK_SIZE>>>(distances, outDist, outIndx);
		
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
	printf("Bandwidth: %f GB/sec\n",
		(DB_SIZE*DIM*4)/((duration.tv_nsec / 1e10)* 1e9));
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
