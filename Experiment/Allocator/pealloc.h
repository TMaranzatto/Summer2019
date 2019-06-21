#pragma once
#include <stdlib.h>


class pealloc {
private: 
	size_t size; 
	int blocks; 




public: 


	pealloc() {
		size = 0; 
		blocks = 0; 


	}

	~pealloc() {



	}

	void* setptr(size_t s) {
		return malloc(s);

	}

	void freenode(void* Node) {
		free(Node);


	}

};






