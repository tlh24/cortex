#ifndef _SIMSEARCH_H_
#define _SIMSEARCH_H_

#define WARPSTEPS 30
#define NUM_BLOCKS 400
#define BLOCK_SIZE 256 // 256 and 512 yield similar bandwidths.
#define BLOCK_STRIDE (BLOCK_SIZE/32)
#define DIM (32 * WARPSTEPS)
#define DIMSHORT 900
#define DB_SIZE (NUM_BLOCKS * BLOCK_SIZE)

typedef struct {
	unsigned char* db;
	unsigned char* query;
	float* distances;
	float* outDist;
	int*   outIndx;
	float* h_outDist;
	int*   h_outIndx;
} imgdb;

#ifdef __cplusplus
extern "C" {
#endif

extern imgdb* simdb_allocate(int num);
extern void simdb_free(imgdb* sdb);
extern void simdb_set(imgdb* sdb, int i, unsigned char* row);
extern void simdb_get(imgdb* sdb, int i, unsigned char* row); 
extern void simdb_query(imgdb* sdb, unsigned char* query,
				float* minDist, int* minIndx);
extern double simdb_checksum(imgdb* sdb); 
extern void simdb_clear(imgdb* sdb); 

#ifdef __cplusplus
}
#endif

#endif
