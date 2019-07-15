#include <cstddef>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct arrayNode{
	int[64] array;
	unsigned long long int bitmap;
	int min;
	int max;
	arrayNode* next;

}

//this implementation assumes a uniform distro of keys
//as well as naiive partitions
//no merging implemented yet
//assume value is within some reasonable range
__global__
void insert(arrayNode *start, int value){
	//finding the location to insert
	arrayNode *current = *start;
	int jumps[resolution] = { 3, 5, 7, 11, 13, 17, 19, 23 };

	while(current->next != NULL){
		

		else if((current->min) <= value && value <= (current->max)){
			for(int i = 0; i < 64; i++){
				if (current->bitmap != ULLONG_MAX){

					int jump = jumps[threadIdx.x % resolution];
					unsigned long long int index = (blockIdx.x * blockDim.x + threadIdx.x + jump) % 64;
					unsigned long long int loc = 1 << i;
					unsigned long long int previousValue = atomicOr(bitmap, loc);

					if ((previousValue >> i) & 1 == 0) {

						current->array[index] = value;
					}
					else{

						continue;
					}
				}
				else{

					continue;
				}
			}
		}
		else{
			*current = current->next;
		}
	}


}

//returns "random" value for threadId%64 if not taken
//else returns NULL
//assumes array is for integers for simplicity
//can template this later
__global__
void get(int* array, unsigned long long int* bitmap) {
	printf("your thread is %d.%d.\n", blockIdx.x * blockDim.x, threadIdx.x);

	const int resolution = 8;
	int jumps[resolution] = { 3, 5, 7, 11, 13, 17, 19, 23 };
	//Loop through all 64 elements to see if we find one that works
	//hopefully our thdId gives us a result immediatly
	for(int k = 0; k < 64; k++){
		int flag = 0;
		int test = 0;
		//checking edge case that the array is empty
		//send a message to host
		if (*bitmap == ULLONG_MAX) {
			//do something
			
			flag = -1;
			#if __CUDA_ARCH__ >= 200
				printf("%d\n", flag);
			#endif
		}
		int jump = jumps[threadIdx.x % resolution];
		unsigned long long int i = (blockIdx.x * blockDim.x + threadIdx.x + jump) % 64;
		unsigned long long int loc = 1 << i;

		unsigned long long int previousValue = atomicOr(bitmap, loc);
		if ((previousValue >> i) & 1 == 0) {
			//do something with the value
			flag = 1;
			#if __CUDA_ARCH__ >= 200
						printf("%d\n", flag);
			#endif
		}

		else{
			//else try next element
			continue;
			#if __CUDA_ARCH__ >= 200
				printf("try again\n");
			#endif
		}

	}
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
	printf("processing...\n");

	get<<<1, 16>>>(arr, bitmap);
	cudaDeviceSynchronize();

	printf("task complete.\n");
	cudaFree(bitmap);
	cudaFree(arr);


}