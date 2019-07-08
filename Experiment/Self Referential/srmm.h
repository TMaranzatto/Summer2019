//Created By Nick Stone 
//Refactored 7/7/2019
//The goal of this program is to incorporate a basic allocator which is self referential 
//Each node holds a pointer to a void* namely these are the address locations 
//This program will use sys mann or MMAP and Munmap virtual pages
//That hold the values that this program allocates.
//This program also frees the memory as well...



#pragma once
#include <iostream>
#include <sys/mman.h>
#include <cstddef>
using namespace std;


class srmm {

	



private:

	struct Node {
		Node* next;
		void* data;
		bool inuse;
		size_t Allocsize;

		//int val;
		//int key; 
		//bool marked;
		bool end;
		Node(Node* nnext, void* id, bool iuse, size_t s, bool iend) {
			this->next = nnext;
			this->data = id;
			this->inuse = iuse;
			this->Allocsize = s;
			//this->key = kkey;
			//this->marked = mmarked;
			this->end = iend;


		}


	};
	//Linked List Data Structure
	Node* head;

	//Size of the heap 
	size_t heap;
	size_t next;



	//Pointer to the first value 

	void* start;

	//Number of Pieces allocated
	int count;


public:

	/*
	Insert Comments
	*/
	srmm() {
		head = NULL;
		heap = 0;
		next = 0;
		start = nullptr;
		count = 0;

		/*
		Instaniate Some of the allocator data here

		*/
	}

	/* 
	Insert Comments
	*/

	~srmm() {
		if (head) {
			Node* lead = head->next;
			Node* follow = head;
			while (lead) {
				delete(follow);
				follow = lead;
				lead = lead->next;
			}
			delete(follow);
		}
	}


	//Node(Node* nnext, T vval, bool mmarked, bool iend) {

	void insert(void* vall, size_t insize) {

		Node* temp = head;
		//Empty list 
		if (head == NULL) {
			head = new Node(NULL, vall, true, insize, true);
		}

		//One element
		else if (head->next == NULL) {
			head->end = false;
			Node* temp = new Node(NULL, vall, true, insize, true);
			head->next = temp;

		}

		//Arbitrary Number of elements
		else {


			while (temp->next != NULL) {
				temp = temp->next; 
				continue;



			}
			//temp->marked = false;
			Node* curr = new Node(NULL, vall, true, insize, true);
			temp->next = curr;
		}
		count = count + 1;
	}
	
	/*
	INsert COmments
	*/
	void remove() {
		//size_t temps = head ->;
		if (head == NULL) {
			cout << "NULL";
		}

		//Only one element
		else if (head->next == NULL) {
			delete(head);
			head = NULL;
		}
		else {
			Node* temp = head->next;
			delete(head);
			head = temp;
		}

		count = count - 1;


	}
	/*
	
	Insert Comments
	*/
	void print() {
		Node* temp = head;

		while (temp != NULL) {
			cout << temp->data;
			cout << endl;
			cout << "Size of " << temp->Allocsize;
			cout << endl;
			cout << endl;
			temp = temp->next;


		}


	}
	/*
	Insert Comments
	
	*/

	bool search(void* value) {
		Node* temp = head;
		//bool marked[1] = { false };
		if (head == NULL) {
			return false;
		}
		while (temp != NULL) {
			if (temp->data == value) {
				cout << "FOUND";
				return true;
			}

			else {
				temp = temp->next;


			}

		}
		return false;
	}
	//END DATA STRUCTURE
	/*
	
	Insert Commments
	*/


	//BEGIN ALLOCATOR
	size_t PAD(size_t s) {
		size_t temp = s;
		if (s % 8 == 0) {
			return temp;

		}
		else {
			temp = s / 8;
			size_t offset = 8 - temp;
			temp = s + offset;
			return temp;

		}

		//return temp; 
	}

	/*
	
	Insert Comments
	*/

	void* malloc(size_t s) {


		Node* temp = head;
		while (temp != NULL) {
			if (temp->inuse == false) {
				if (temp->Allocsize == s) {
					temp->inuse = true; 
					return temp->data;


				}

			}
			temp = temp->next;

		}

		void* temps = maps(s);

		return temps;

	}
	/*
	
	Insert Comments
	
	*/
	
	void* map(size_t len) {
		void* temp;
		temp = mmap(nullptr,
			len,
			PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1,
			0);
		return temp;


	}
	/*
	
	Insert Comments
	*/

	void* maps(size_t f) {
		void* temp;
		temp = mmap(nullptr,
			f,
			PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1,
			0);
		return temp;

	}



	/*
	
	Insert Comments

	*/

	void free(void * stemp) {

		Node* temp = head;
		if (head == NULL) {
			cout << "List is empty nothing to free";

		}
		else {

			while (temp != NULL) {
				if (temp->data == stemp) {
					temp->inuse = false;
					cout << "WE FOUND IT " << endl;
					break;
				}
				else {

					temp = temp->next;
				}
			}

		}



	}

};