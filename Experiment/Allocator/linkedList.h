//Created by Nick Stone and Thomas Maranzatto 

//Concurrent single linked link list. 

//Assuming all values are unique

//Created on 6/17/2019



#pragma once
#include <iostream>
#include <cstddef>
#include "salloc.h"
using namespace std;

#ifndef linkedList_H_
#define linkedList_H_

#endif


//template <class A> 

class linkedList {

private:

	struct Node {
		void* next;
		int val;
		//int key; 
		//bool marked;
		bool end;
		Node(void* nnext, int vval, bool iend) {
			this->next = nnext;
			this->val = vval;
			//this->key = kkey;
			//this->marked = mmarked;
			this->end = iend;


		}


	};
	void* head;
	salloc s; 



public:

	linkedList() {
		head = NULL;
		
	}

	~linkedList() {
		if (head) {
			void* lead = head->next;
			void* follow = head;
			while (lead) {
				s.freeNode(follow);
				
				follow = lead;
				lead = lead->next;
			}
			
			s.freeNode(follow);
		}
	}


	//Node(Node* nnext, T vval, bool mmarked, bool iend) {

	void insert(int vall) {

		void* temp = head;
		//Empty list 
		if (head == NULL) {
			
			
			head = new Node(NULL, vall,  true);
		}

		//One element
		else if (head->next == NULL) {
			head->end = false;
			void * temp = s.createNode(sizeof(struct Node(NULL,vall,true)));
			//Node* t = &temp;


			//Points the memory address to the Location of the given Node 
			//THE only error is because Void * != Node 

			//Finding a way to cast next

			temp = Node l(NULL, vall, true);
			
			
			
			//Node l(NULL,vall,true); 

			//Node* temp = new Node(NULL, vall,  true);
			head->next = temp;

		}

		//Arbitrary Number of elements
		else {


			while (temp->next != NULL) {
				continue;



			}
			//temp->marked = false;
			void* curr = new Node(NULL, vall, true);
			temp->next = curr;
		}

	}
	/*
	void mark(int vall, bool mmar) {
		if (head == NULL) {
			cout << "List is empty";
			//break;
		}
		Node* temp = head;
		while (temp->next != NULL) {
			if (temp->val == vall && temp->marked == mmar) {
				cout << "Already IN this state";
				break;

			}
			else if (temp->val == vall && temp->marked != mmar) {
				temp->marked = mmar;

			}
			else {
				temp = temp->next;

			}

		}

	}*/
	void remove() {
		if (head == NULL) {
			cout << "NULL";
		}

		//Only one element
		else if (head->next == NULL) {
			s.freeNode(head);
			head = NULL;
		}
		else {
			void* temp = head->next;
			s.freeNode(head);
			
			head = temp;
		}



	}

	void print() {
		void* temp = head; 
		
		while (temp != NULL) {
			cout << temp->val;
			cout << endl; 
			temp = temp->next; 


		}


	}

	bool search(int value) {
		void* temp = head;
		//bool marked[1] = { false };
		if (head == NULL) {
			return false;
		}
		while (temp != NULL) {
			if (temp->val == value) {
				cout << "FOUND";
				return true;
			}
			
			else {
				temp = temp->next;


			}

		}
		return false;
	}



};