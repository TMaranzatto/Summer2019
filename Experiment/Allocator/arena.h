#pragma once
#include <iostream>
using namespace std; 



class arena {
private: 
	void* start; 

	size_t heapsi; 

	//size_t next; 

	struct Node {
		Node* next; 
		size_t nextal; 
		size_t allocated; 
		void* where; 

		Node(Node* inext, size_t inextal, size_t iallocated, void* iwhere) {
			this->next = inext; 
			this->nextal = inextal; 
			this->allocated = iallocated; 
			this->where = iwhere; 


		}
	

	};

	Node* head; 



public: 

	arena() {

		head = NULL; 
		start = NULL; 
		heapsi = 0; 

		
	}
	~arena() {


	}


	void* malloc(size_t s) {

		void* temp = malloc(s);

		return temp; 



	}


	void free() {

		//No Op 

	}








};



