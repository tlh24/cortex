#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "simsearch.h"

// Function to compute the difference between two timespec values
struct timespec diff_timespec(struct timespec start, struct timespec end) {
    struct timespec diff;

    // If the nanoseconds value of 'end' is smaller
    // it borrows 1 second from the seconds value
    if (end.tv_nsec < start.tv_nsec) {
        diff.tv_sec = end.tv_sec - start.tv_sec - 1;
        diff.tv_nsec = end.tv_nsec - start.tv_nsec + 1E9;  // 1E9 == 1,000,000,000
    } else {
        diff.tv_sec = end.tv_sec - start.tv_sec;
        diff.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return diff;
}

int main()
{
	printf("num blocks: %d db_size:%d, %f MB\n", NUM_BLOCKS, DB_SIZE,
			 (float)(DB_SIZE * DIMSHORT) / 1e6);

	imgdb* sdb = simdb_allocate(DB_SIZE);

	// Choose a random index
	srand(time(NULL));   // Initialization, should only be called once.
	int random_indx = rand() % DB_SIZE;

	// fill the DB with random uchars
	unsigned char* q = (unsigned char*)malloc(DIMSHORT);
	for(int i=0; i<DB_SIZE; i++){
		for(int j=0; j<DIMSHORT; j++){
			q[j] = (unsigned char)(rand() % 256); 
		}
		simdb_set(sdb, i, q);
	}

	for(int j=0; j<DIMSHORT; j++){
		q[j] = (unsigned char)(rand() % 256); 
	}
	simdb_set(sdb, random_indx, q);
	
	struct timespec time1, time2;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);

	float minDist;
	int minIndx;
	for(int u = 0; u < 20; u++){
		simdb_query(sdb, q, &minDist, &minIndx);
	}
	
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
	struct timespec duration = diff_timespec(time1, time2);
	
	printf("Execution time: %f Bandwidth: %f GB/sec\n",
		(duration.tv_nsec / 2e10), 
		(DB_SIZE*DIM)/((duration.tv_nsec / 2e10)* 1e9));

	printf("Best match: %d, should be %d; Minimum distance: %f\n",
			 minIndx, random_indx, minDist);

	simdb_free(sdb);
	return 0;
}
