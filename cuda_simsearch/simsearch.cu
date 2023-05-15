#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

#define WARPSTEPS 30
#define NUM_BLOCKS 400
#define BLOCK_SIZE 256 // 256 and 512 yield similar bandwidths.
#define BLOCK_STRIDE (BLOCK_SIZE/32)
#define DIM (32 * WARPSTEPS)
#define DIMSHORT DIM
#define DB_SIZE (NUM_BLOCKS * BLOCK_SIZE)

using namespace std;

// this (w/ reduce) takes 4.2ms on 3080 laptop.
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

struct imgdb {
	unsigned char* db;
	unsigned char* query;
	float* distances;
	float* outDist;
	int*   outIndx;
	float* h_outDist;
	int*   h_outIndx;
};

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

int main()
{
	printf("num blocks: %d db_size:%d\n", NUM_BLOCKS, DB_SIZE);

	imgdb* sdb = simdb_allocate(DB_SIZE);

	// Choose a random index
	std::random_device rd;
	std::mt19937 generator(rd());

	// fill the DB with random uchars
	unsigned char* q = (unsigned char*)malloc(DIMSHORT);
	std::uniform_int_distribution<int> distribution256(1, 256);
	for(int i=0; i<DB_SIZE; i++){
		for(int j=0; j<DIMSHORT; j++){
			q[j] = (unsigned char)distribution256(generator);
		}
		simdb_set(sdb, i, q);
	}

	std::uniform_int_distribution<int> distribution(1, DB_SIZE);
	int random_indx = distribution(generator);

	for(int j=0; j<DIMSHORT; j++){
		q[j] = (unsigned char)distribution256(generator);
	}
	simdb_set(sdb, random_indx, q);

	timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

	float minDist;
	int minIndx;
	for(int u = 0; u < 10; u++){
		simdb_query(sdb, q, &minDist, &minIndx);
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

	timespec duration = diff(time1, time2);
	cout<<duration.tv_nsec / 1e10 <<endl;
	printf("Bandwidth: %f GB/sec\n",
		(DB_SIZE*DIM)/((duration.tv_nsec / 1e10)* 1e9));
	printf("Best match: %d, should be %d; Minimum distance: %f\n",
			 minIndx, random_indx, minDist);

	simdb_free(sdb);
	return 0;
}

// int main() {
// 	unsigned char* db;
// 	unsigned char* query;
// 	float* distances;
// 	float* outDist;
// 	int*   outIndx;
//
// 	int num_blocks = NUM_BLOCKS;
// 	printf("num blocks: %d\n", num_blocks);
//
// 	imgdb* sdb = simdb_allocate(DB_SIZE);
//
//
//
// 	// Allocate memory for the database, query vector, distances, and best match index
// 	unsigned char* hdb = (unsigned char*)malloc(DB_SIZE * DIM);
// 	cudaMalloc(&db, DB_SIZE * DIM * sizeof(unsigned char));
// 	cudaMalloc(&query, DIM * sizeof(float));
// 	cudaMalloc(&distances, DB_SIZE * sizeof(float));
// 	cudaMalloc(&outDist, num_blocks * sizeof(int));
// 	cudaMalloc(&outIndx, num_blocks * sizeof(float));
//
//
// 	// Choose a random index
// 	std::random_device rd;
// 	std::mt19937 generator(rd());
//
// 	// Now generate random numbers with generator
// 	std::uniform_int_distribution<int> distribution(1, DB_SIZE);
// 	int random_indx = distribution(generator);
//
// 	// fill the DB with random uchars
// 	std::uniform_int_distribution<int> distribution256(1, 256);
// 	for(int i=0; i<DB_SIZE * DIM; i++){
// 		hdb[i] = (unsigned char)distribution256(generator);
// 	}
// 	cudaMemcpy(db, hdb, DB_SIZE*DIM, cudaMemcpyHostToDevice);
//
// 	// Get the query vector from the random index
// 	cudaMemcpy(query, db + random_indx * DIM, DIM, cudaMemcpyDeviceToDevice);
//
// 	float* h_outDist = (float*)malloc(num_blocks*sizeof(float));
// 	int* h_outIndx = (int*)malloc(num_blocks*sizeof(int));
// 	float minDist;
// 	int minIndx;
// 	dim3 dimBlock(32, BLOCK_STRIDE, 1); // x, y, z
//
// 	timespec time1, time2;
// 	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
//
// 	for(int u = 0; u < 10; u++){
// 		// Compute the L2 distances
// 		//compute_distances<<<num_blocks, 256>>>(db, query, distances);
// 		compute_distances2<<<DB_SIZE/BLOCK_STRIDE, dimBlock>>>(db, query, distances);
//
// 		findMinOfArray<<<num_blocks, BLOCK_SIZE>>>(distances, outDist, outIndx);
//
// 		cudaMemcpy(h_outDist, outDist, sizeof(float)*num_blocks, cudaMemcpyDeviceToHost);
// 		cudaMemcpy(h_outIndx, outIndx, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost);
//
// 		minDist = h_outDist[0];
// 		minIndx = h_outIndx[0];
// 		for(int i=1; i<num_blocks; i++){
// 			if(h_outDist[i] < minDist){
// 				minDist = h_outDist[i];
// 				minIndx = h_outIndx[i];
// 			}
// 		}
// 	}
// 	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
//
// 	timespec duration = diff(time1, time2);
// 	cout<<duration.tv_nsec / 1e10 <<endl;
// 	printf("Bandwidth: %f GB/sec\n",
// 		(DB_SIZE*DIM)/((duration.tv_nsec / 1e10)* 1e9));
// 	printf("Best match: %d, should be %d; Minimum distance: %f\n",
// 			 minIndx, random_indx, minDist);
//
// 	// Free the memory
// 	cudaFree(db);
// 	cudaFree(query);
// 	cudaFree(distances);
// 	cudaFree(outDist);
// 	cudaFree(outIndx);
// 	free(h_outDist);
// 	free(h_outIndx);
//
// 	return 0;
// }
