#pragma once
#include <cstddef>



class Simpleallocator {

private: 


	struct freenode {

		struct freenode* next; 

	};
	freenode* next_store = nullptr; 
	static const int num_chunk = 4; 


public: 
	void* allocate(size_t);
	void deallocate(void*, size_t);






};


