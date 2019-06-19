#pragma once
#include <stdlib.h>
#include <iostream>
//#include <unistd.h>

class Simpleallocator {
private: 

	struct Node {
		//Is this free
		bool free; 
		//Size of Block
		size_t size; 
		//What memory block is next
		Node* next; 
		void* block; 
		Node(bool ifree, size_t isize, Node* inext, void* iblock) {
			this ->free = false; 
			this ->size = isize; 
			this ->next = inext; 
			this ->block = iblock;


		}

	};

	Node* head; 
	Node* tail; 
	int max; 
	int offset; 
	



public: 


	Simpleallocator(int regionmax) {
		max = regionmax; 
		offset = 8; 

	}

	~Simpleallocator() {



	}
	
	void* mmalloc(size_t isize) {
		void* block; 
		//block = sbrk(isize;


	}

	void free(void* block) {


	}

	void set_heap(size_t size) {


			

	}

	void freenode() {


	}


};