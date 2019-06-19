#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <ctime>

__global__ 
void dflip(bool* ref){
    *ref = !(*ref);
}

__host__ 
void hflip(bool* ref){
	*ref = !(*ref);
}

int main(){
    bool* flipBit;
    cudaMallocManaged(&flipBit, sizeof(bool));
	*flipBit = true;

	std::clock_t start;

	double totalTime = 0;
	int rounds = 100;

	for (int i = 0; i < rounds; i++) {
		//start timer
		
		double duration;
		start = std::clock();

		for (int j = 0; j < 1000000; j++) {
			dflip <<<1, 1 >>> (flipBit);
			hflip(flipBit);
		}


		//end timer and print result
		totalTime += (std::clock() - start) / (double)CLOCKS_PER_SEC;
		
	}
	cudaFree(flipBit);
	std::cout << "average duration for 1 mil. bitflips over " << rounds <<" runs: " << totalTime/rounds << '\n';
    return 0;
}