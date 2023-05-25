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
	printf("Execution time: %f Bandwidth: %f GB/sec\n",
		(duration.tv_nsec / 2e10), 
		(DB_SIZE*DIM)/((duration.tv_nsec / 2e10)* 1e9));
	printf("Best match: %d, should be %d; Minimum distance: %f\n",
			 minIndx, random_indx, minDist);
	
	// check clearing & maximum distance
	for(int u=1; u < DIMSHORT; u++){
		simdb_clear(sdb); // all zeros. 
		int j = 0; 
		for(; j<u; j++){
			q[j] = (unsigned char)255;
		}
		for(; j<DIMSHORT; j++){
			q[j] = (unsigned char)0;
		}
		simdb_query(sdb, q, &minDist, &minIndx);
		int y = round(minDist / (255.0 * 255.0)); 
		if(u != y) { 
			printf("defined dist: %d, should be %d\n", y, u); 
		}
	}
	
	//check random distance
	simdb_clear(sdb);
	float sum = 0.0; 
	for(int j=0; j<DIMSHORT; j++){
		int u = distribution256(generator);
		q[j] = (unsigned char)u;
		sum += u * u; 
	}
	simdb_query(sdb, q, &minDist, &minIndx);
	printf("defined dist: %f, should be %f\n",
				minDist , sum);

	simdb_free(sdb);
	return 0;
}
