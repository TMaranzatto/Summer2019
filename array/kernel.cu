#include <cstddef>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>

struct arrayNode {
	int array[64];
	unsigned long long int bitmap;
	int min;
	int max;
	arrayNode* next;
	int seqLock;
	bool isStart;

	arrayNode(int minn, int maxx, arrayNode* nextt){
		bitmap = 0;
		min = minn;
		max = maxx;
		next = nextt;
		seqLock = 0;
		isStart = false;
	}
	~arrayNode(){}
};

//this implementation assumes a uniform distro of keys
//as well as naiive partitions
//no merging implemented yet
//assume that our value is within some reasonable range
__global__
void insert(arrayNode *start, int value){
	startInsert:
	//finding the location to insert
	arrayNode *current = start;
	//while we are not in the 
	while ((current->next)->next != NULL) {
		//1. find location to insert into, and skip the first dummy node
		if (current->min <= value && current->max >= value && current->isStart == true) {
			int insertion_location = (int)(value / ((current->max - current->min) / 64));
			int dir = -1;
			bool useCurrDir = false;
			bool successful_insertion_flag = false;

		//2. try to insert at the location in the array the value
		//would ideally exist in
			for (int i = 0; i < 64; i++) {
				int search_location = insertion_location + dir * i;
				insertion_location = search_location;
				unsigned long long int bitmap_location = 1 << search_location;

				//checking if we are outside bounds
				if (search_location < 0 || search_location >= 64) { break; }

				//checking if we are above the target value
				//which would break the sorting situation

				if (dir == -1) {
					if (current->array[search_location] > value && useCurrDir == false) { 
						dir *= -1; 
						useCurrDir = true; 
						continue;
					}
					else if (current->array[search_location] > value && useCurrDir == true){
						break;
					}
				}

				//and checking for below target
				else{
					if (current->array[search_location] < value && useCurrDir == false) { 
					dir *= -1; 
					useCurrDir = true; 
					continue;
					}
					else if (current->array[search_location] < value && useCurrDir == true){
						break;
					}
				}

				//if we succeed all the above value and boundary conditions
				//then we try to modify data in the array through the proxy
				//of our bitmap
				unsigned long long int previousValue = atomicOr(current->bitmap, bitmap_location);
				if ((previousValue >> bitmap_location) & 1 == 0) {
					current->array[search_location] = value;
					//setting flag for safety.
					//could get rid of this in refactoring
					successful_insertion_flag = true;
					return;
				}

				//otherwise, lets loop again and hope it works
				else {
					if( useCurrDir == false){ dir *= -1; }
					continue;
				}
			}

		//3. if above fails, do the split routine
			if (successful_insertion_flag == false) {
				if(current->seqLock%2 == 1){
					goto startInsert;
				}
				atomicCAS(current->seqLock)
				//NEED LOCK HERE
				arrayNode *new_arrayNode = new arrayNode(-1,-1, NULL);
				//very slow here but should be working
				//need to speed this up later

				//setting new array values
				int minval = 10000000;
				int maxval = 0;
				for (int i = 0; i < 32; i++) {
					int new_value = current->array[32 + i]
					if(new_value != 0){
						new_arrayNode->array[2 * i] = new_value;
						new_arrayNode->bitmap |= unsigned long long int (1<<(2 * i));
					}
					minval = min(minval, new_value)
					maxval = max(minval, cnew_value)
					//new_arrayNode->array[(2 * i) + 1] = current->array[32 + i];
				}
				new_arrayNode->min = minval;
				new_arrayNode->max = maxval;

				//and old array values
				minval = maxval + 1;
				maxval = 0;
				for (int i = 31; i >= 0; i--) {
					int new_value = current->array[i]
					if(new_value != 0){
						new_arrayNode->array[2 * i + 1] = new_value;
						new_arrayNode->bitmap |= unsigned long long int (1<<(2 * i + 1));
					}
					current->array[2 * i + 1] = current->array[i];
					maxval = max(minval, current->array[i])
				}
				current->min = minval;
				current->max = maxval;
				
				//finally setting the node pointers 
				new_arrayNode->next = current->next;
				current->next = new_arrayNode;
			}
		}
		//TODO
		//4. else, if this array and its right neighbor are < third full
		//merge them
		
		else {
			current = current->next;
			continue;
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