//Created By Nick Stone 
//Refactored 7/6/2019
//The goal of this program is to incorporate a basic allocator which is self referential 
//Each node holds a pointer to a void* namely these are the address locations 
//That hold the values that this program allocates.
//This program also frees the memory as well...

#pragma once
#include <iostream>
#include <cstddef>
#include <vector>
using namespace std;


class srmap {



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

	srmap() {
		head = NULL;
		heap = 0;
		next = 0;
		start = nullptr;
		count = 0;

		/*
		Instaniate Some of the allocator data here

		*/
	}

	~srmap() {
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

	void print() {
		Node* temp = head;

		while (temp != NULL) {
			cout << temp->data;
			//cout << endl; 
			cout << " Size of " << temp->Allocsize;
			//cout << endl; 
			cout << endl;
			temp = temp->next;


		}


	}

	vector<size_t> sizeprint() {
		
		Node* temp = head;
		vector<size_t> sizes; 
		while (temp != NULL) {
			cout << temp->data;
			//cout << endl; 
			sizes.push_back(temp->Allocsize);
			cout << " Size of " << temp->Allocsize;
			//cout << endl; 
			cout << endl;
			temp = temp->next;


		}
		return sizes; 


	}

	vector<void*> voidprint() {

		Node* temp = head;
		vector<void*>voidss;
		while (temp != NULL) {
			cout << temp->data;
			//cout << endl; 
			voidss.push_back(temp->data);
			cout << " Size of " << temp->Allocsize;
			//cout << endl; 
			cout << endl;
			temp = temp->next;


		}
		return voidss;


	}





	bool search(void* value) {
		Node* temp = head;
		
		if (head == NULL) {
			return false;
		}
		while (temp != NULL) {
			if (temp->data == value) {
				
				return true;
			}

			else {
				temp = temp->next;


			}

		}
		return false;
	}

	void* malloc(size_t s) {
		/*
		The following is logic to identify is memory has already been allocated and is
		not needed. IE just overwrite the node and call it in use!


		*/
		Node* temph = head;

		if (head == NULL) {

		}

		while (temph != NULL) {
			if (temph->inuse == false) {
				if (temph->Allocsize == s) {
					
					temph->inuse = true;
					return temph->data;


				}


				else {
					temph = temph->next;
				}
			}


			else {
				temph = temph->next;
				
				
			}

		}


		size_t pad_size = PAD(s);        // TODO: pad the size appropriately
		void* retval = &start + next;
		next += pad_size;
		heap += pad_size;
		return retval;
	}

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

	int free(void* stemp) {
		Node* temp = head;
		if (head == NULL) {
			cout << "List is empty nothing to free";
			return 0;
		}
		else {

			while (temp != NULL) {
				if (temp->data == stemp) {
					temp->inuse = false;
				
				 
					return 0;
				}
				else {

					temp = temp->next;
				}

			}
		
		}
		return 0;
	}

};