#include <stdio.h>
#include <iostream>
#include <random>
#include "simsearch.h"

using namespace std;

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

int main()
{
	printf("num blocks: %d db_size:%d, %f MB\n", NUM_BLOCKS, DB_SIZE,
			 (float)(DB_SIZE * DIMSHORT) / 1e6);

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
	for(int u = 0; u < 20; u++){
		simdb_query(sdb, q, &minDist, &minIndx);
	}

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);

	timespec duration = diff(time1, time2);
	cout<<duration.tv_nsec / 2e10 <<endl;
	printf("Bandwidth: %f GB/sec\n",
		(DB_SIZE*DIM)/((duration.tv_nsec / 2e10)* 1e9));
	printf("Best match: %d, should be %d; Minimum distance: %f\n",
			 minIndx, random_indx, minDist);

	simdb_free(sdb);
	return 0;
}
