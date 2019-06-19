#pragma once
#pragma once
#include <iostream>
#include <cstddef>
#include <atomic>


using namespace std;

#ifndef ccll_H_
#define ccll_H_
#endif


class ccll {
private: 
	struct Node {
		Node* next;
		int val;
		//int key; 
		//bool marked;
		bool end;
		//bool lock;
		Node(Node* nnext, int vval, bool iend) {
			this->next = nnext;
			this->val = vval;
			//this->key = kkey;
			//this->marked = mmarked;
			this->end = iend;
			//this->lock = ilock; 


		}


	};
	Node* head;



public: 

	ccll() {
		head = NULL;
	}

	~ccll() {
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



	int remove(int val) {
		//Make Dummy Variables
		Node* left; 
		Node* right; 
		Node* curr; 


		while (1) {
			
			


		}




	}

	int insert(int vall)
	{
		//make Dummy Variables
		Node* temp = head; 
		while (1) {
			if (head == NULL) {
				head = new Node(NULL, vall, true);
				//ATOMIC CAS
				if () {
					cout << "success";
					return 1;
				}
				else {

				}
				//Node Atomic_exchange(temp, new Node(NULL, vall, true));

			}
			else {
				while (temp-> next != NULL) {
					temp = temp->next; 

				}
				Node* nn = new Node(NULL, vall , true);
				temp->end = false; 
				temp->next = nn; 
				//ATOMIC CAS
				if () {
					cout << "success";
					return 1; 
				}
				else {

				}
			
			}
		}


	}





	bool search(int value) {
		Node* temp = head;
		//bool marked[1] = { false };
		if (head == NULL) {
			return false;
		}
		while (1) {

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

		
	}

	/*
	THis FUNCTION IS TO ONLY BE USED FOR SEQUENTIAL TESTING

	Planning on using this to ensure that the code is correct once mutliple threads run against this LL
	
	
	
	*/
	void print() {
		Node* temp = head;

		while (temp != NULL) {
			cout << temp->val;
			cout << endl;
			temp = temp->next;


		}


	}











};