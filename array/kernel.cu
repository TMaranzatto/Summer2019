#include <cstddef>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <algorithm>


//defining our structs for the data structure

//this is a simple key value pair with constructor
template<typename KEY, typename VALUE>
struct KeyValue{
public:
	KEY key;
	VALUE value;

	KeyValue(KEY kkey, VALUE vval) {
		key = kkey;
		value = vval;
	}
};

//this is the arrayNode, the key structure we operate over for our graph search
//becuase the above is templated, we define clearly what our data types will be for the graph
//(TODO) it holds an array of key value pairs as well as a bitmap denoting which indices have data
//two locks indicate if we are modifying internal (array) data or external (next pointer) data
struct arrayNode {
public:
	//KeyValue array[64]
	int array[64];
	int min;
	int max;
	arrayNode* next;
	bool isStart;

	unsigned long long int bitmap;
	int internal_lock;
	int external_lock;
	int operating_threads;

	arrayNode(void){
		isStart = false;
		external_lock = 0;
		internal_lock = 0;
		operating_threads = 0;
		}
};

//this implementation assumes a uniform distro of keys
//as well as naiive partitions
//no merging implemented yet
//assume that our value is within some reasonable range
__global__
void insert(arrayNode *start, int value){

	startInsert:
	arrayNode *current = start;
	while ((current->next)->next != NULL) {
//CASE 1: find location to insert into, and skip the first dummy node
//this uses a dead simple traversal technique for a linked list

		//if this node is being externally modified we should avoid modifying it for now
		if (current->external_lock == 1) { goto startInsert; }
		if (current->min <= value && current->max >= value && current->isStart == true) {

			//as we are operating in this node,  set the internal_lock flag to true
			//as well as increase the working threads node variable
			atomicCAS(current->internal_lock, 0, 1);
			atomicAdd(current->operating_threads, 1);

			int insertion_location = (int)(value / ((current->max - current->min) / 64));
			int dir = -1;
			bool useCurrDir = false;
			bool successful_insertion_flag = false;

//CASE 2: try to insert at the ideal location in the array.  If this fails we
//'bounce' around the value, checking to see if there are available slots on the
//left or right side of the ideal location to insert.  If this too fails, we move to
//the next case
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

				//otherwise, lets loop again and hope it works for the next index in the array
				else {
					if( useCurrDir == false){ dir *= -1; }
					continue;
				}
			}

//TODO: MAKE THE BELOW OPERATE IN THIS CASE AS WELL AS THE 2/3 FULL CASE

//CASE 3: if above fails, call the 'split' routine.  This seperates the data in the arraynode
// into two new arraynodes with the data spaced out evenly.  This involves some complex locking
//mechanisms and waiting, but this (should) work in a reasonable amount of time.
			if (successful_insertion_flag == false) {
				//the internal working thread is converted into an external working thread
				//and so the number of internal working threads must decrease
				//but only after we declare that we are working on this node to the data struct.
				//else we get weird edge cases where multiple threads are trying to insert
				if (atomicCAS(current->external_lock, 0, 1) == 1) {
					goto startInsert;
				}
				atomicSub(current->operating_threads, 1);

				//now we spin until the number of active internal threads in the node
				//reaches 0.  this is wasteful, but hopefully only a handful of threads are working
				//on a node at any point, and the above insertion operation is quick
				while(current->internal_lock == 1){
					continue;
				}

				arrayNode *new_arrayNode = new arrayNode();

				//setting new array values
				int minval = 10000000;
				int maxval = 0;
				for (int i = 0; i < 32; i++) {
					int new_value = current->array[32 + i];
					if(new_value != 0){
						new_arrayNode->array[2 * i] = new_value;
						new_arrayNode->bitmap |= unsigned long long int (1<<(2 * i));
					}
					minval = std::min(minval, new_value);
					maxval = std::max(minval, new_value);
					//new_arrayNode->array[(2 * i) + 1] = current->array[32 + i];
				}
				new_arrayNode->min = minval;
				new_arrayNode->max = maxval;

				//and old array values
				minval = maxval + 1;
				maxval = 0;
				for (int i = 31; i >= 0; i--) {
					int new_value = current->array[i];
					if(new_value != 0){
						new_arrayNode->array[2 * i + 1] = new_value;
						new_arrayNode->bitmap |= unsigned long long int (1<<(2 * i + 1));
					}
					current->array[2 * i + 1] = current->array[i];
					maxval = std::max(minval, current->array[i]);
				}
				current->min = minval;
				current->max = maxval;
				
				//finally setting the node pointers 
				new_arrayNode->next = current->next;
				current->next = new_arrayNode;

				//and now reset our insertion routine and unlock
				current->internal_lock = 0;
				current->external_lock = 0;
				goto startInsert;
			}
		}
		//TODO
		//4. else, if this array and its right neighbor are < 1/3 full merge them
		
		//if our initial question as to if this node contains our desired key range fails
		//then try the next one
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
KeyValue<int, int> get(arrayNode * node){
	const int resolution = 8;
	int jumps[resolution] = { 3, 5, 7, 11, 13, 17, 19, 23 };

	//Loop through all 64 elements to see if we find one that works
	//hopefully our thdId gives us a result immediatly
	for(int k = 0; k < 64; k++){
		int flag = 0;
		int test = 0;
		//checking edge case that the array is empty
		//send a message to host
		if (node->bitmap == ULLONG_MAX) {
			return KeyValue<int, int>(-1, -1);
		}

		int jump = jumps[threadIdx.x % resolution];
		unsigned long long int i = (blockIdx.x * blockDim.x + threadIdx.x + jump) % 64;
		unsigned long long int loc = 1 << i;

		unsigned long long int previousValue = atomicOr(node->bitmap, loc);
		if ((previousValue >> i) & 1 == 0) {
			//do something with the value
			return node->array[i];
		}

		else{
			//else try next element
			continue;
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