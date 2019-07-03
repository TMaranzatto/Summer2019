#include <cstddef>;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//returns value for threadId%64 if not taken
//else returns NULL
//assumes array is for integers
__global__
void get(int* array, unsigned long long int* bitmap) {

	int flag = 0;
	int test = 0;
	//checking edge case that the array is empty
	//send a message to host
	if (*bitmap == ULLONG_MAX) {
		//do something
		flag = -1;
	}

	unsigned long long int i = (blockIdx.x * blockDim.x + threadIdx.x) % 64;
	unsigned long long int loc = 1 << i;

	unsigned long long int previousValue = atomicOr(&bitmap, loc);
	if ((previousValue >> i) & 1 == 0) {
		//do something with the value
		flag = 1;
	}
	//else do nothing
}

int main(void) {
	unsigned long long int *bitmap;
	int *arr;
	cudaMallocManaged(&bitmap, sizeof(unsigned long long int));
	cudaMallocManaged(&arr, 64*sizeof(int));

	//dumb init of array
	for (int i = 0; i < 64; i++) {
		arr[i] = i;
	}

	get <<<1, 256 >>> (arr, bitmap);


}