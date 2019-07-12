#include <cstddef>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

//returns value for threadId%64 if not taken
//else returns NULL
//assumes array is for integers for simplicity
//can template this later

struct arrayNode {
	int data[64];
	unsigned long long int bitmap;
	int lo;
	int hi;
	arrayNode *next;
};
__global__
void get(arrayNode *k){
	unsigned long long int bitmap = k->bitmap;
	int *arr = k->data;

	//defining primes to prevent locking/atomics issues
	const int resolution = 8;
	int jumps[resolution] = { 3, 5, 7, 11, 13, 17, 19, 23 };

	//Loop through all 64 elements to see if we find one that works
	//hopefully our thdId gives us a result immediatly
	for(int k = 0; k < 64; k++){
		int flag = 0;
		//checking edge case that the array is empty
		//send a message to host
		if (bitmap == ULLONG_MAX) {
			//do something
			flag = -1;

		}
		int jump = jumps[threadIdx.x % resolution];
		unsigned long long int i = (blockIdx.x * blockDim.x + threadIdx.x + jump) % 64;
		unsigned long long int loc = 1 << i;

		unsigned long long int previousValue = atomicOr(bitmap, loc);
		if ((previousValue >> i) & 1 == 0) {
			//do something with the value
			flag = 1;

		}

		else{
			//else try next element

			continue;
		}

	}
}

int main(void) {
	arrayNode *k;
	cudaMallocManaged(&k, sizeof(k));

	//dumb init of array
	for (int i = 0; i < 64; i++) {
		k->data[i] = i;
	}

	get<<<1, 256 >>>(k);

	cudaFree(k);

}