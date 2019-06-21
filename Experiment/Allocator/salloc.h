#pragma once
#include <iostream>
#include <stdlib.h>


using namespace std; 


class salloc{

private:
	
	int blocks; 
	size_t size; 
	//free List
	//allocated List 



public:


	salloc() {
		blocks = 0; 
		size = 0; 
	//	int s = 5; 
		//void* t = &s; 

	}

	~salloc() {
		

	}

	void* createNode(size_t s)
	{

		void* n = malloc(s);
		return n; 

	}

	void freeNode(void* Node) {

		free(Node);
		cout << "Node Freed";

	}

	void SetHeap() {


	}

	void* Createptr(size_t size) {
		
		//Take in a size and return the pointer to the storage in memory 
		return malloc(size);


	}


};



