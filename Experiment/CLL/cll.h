//Created by Nick Stone and Thomas Maranzatto 

//Concurrent single linked link list. 

//Assuming all values are unique

//Created on 6/17/2019



#pragma once
#include <iostream>
#include <cstddef>
using namespace std; 



template <class t> class cll {

private:
	
	struct Node {
		Node* next; 
		T val; 
		//int key; 
		bool marked; 
		Node(Node* nnext, T vval, bool mmarked) {
			this->next = nnext; 
			this->val = vval; 
			this->key = kkey; 
			this->marked = mmarked;


		}


	};
	Node* head; 



public:

	cll() {
		head = NULL; 
	}

	~cll() {
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

	}

	void insert(T val) {

		Node* temp = head; 

		while (temp->next == NULL) {


		}


	}

	void mark(T vall, bool mmar) {
		if (head == NULL) {
			cout << "List is empty";
			break; 
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

	}
	void remove() {
		if (head == NULL) {
			cout << "NULL";
		}

		//Only one element
		else if (head->next == NULL) {
			delete(head);
			head = NULL; 
		}
		else {
			Node* temp = head -> next; 
			delete(head);
			head = temp; 
		}

		

	}

	bool search(T val) {
		Node* temp = head; 
		//bool marked[1] = { false };
		if (head == NULL) {
			return false; 
		}
		while (temp -> next!= NULL) {
			if (temp->val == value && temp->marked == false) {
				return true; 
			}
			else if (temp->val == vale && temp->marked == true) {
				return false;
			}
			else {
				temp = temp->next; 


			}

		}
		return false; 
	}



};