//Created by Nick Stone and Thomas Maranzatto 

//Concurrent single linked link list. 

//Assuming all values are unique

//Created on 6/17/2019



#pragma once
#include <iostream>
#include <cstddef>
using namespace std;

#ifndef cll_H_
#define cll_H_

#endif



template <class t> class cll {

private:

	struct Node {
		Node* next;
		int val;
		//int key; 
		bool marked;
		bool end;
		Node(Node* nnext, int vval, bool mmarked, bool iend) {
			this->next = nnext;
			this->val = vval;
			//this->key = kkey;
			this->marked = mmarked;
			this->end = iend;


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


	//Node(Node* nnext, T vval, bool mmarked, bool iend) {

	void insert(int vall) {

		Node* temp = head;
		//Empty list 
		if (head == NULL) {
			head = new Node(NULL, vall, false, true);
		}

		//One element
		else if (head->next == NULL) {
			head->end = false;
			Node* temp = new Node(NULL, vall, false, true);
				head->next = temp;

		}

		//Arbitrary Number of elements
		else {


			while (temp->next != NULL) {
				continue;



			}
			temp->marked = false;
			Node* curr = new Node(NULL, vall, false, true);
			temp->next = curr;
		}

	}

	void mark(int vall, bool mmar) {
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
			Node* temp = head->next;
			delete(head);
			head = temp;
		}



	}

	bool search(int value) {
		Node* temp = head;
		//bool marked[1] = { false };
		if (head == NULL) {
			return false;
		}
		while (temp->next != NULL) {
			if (temp->val == value && temp->marked == false) {
				return true;
			}
			else if (temp->val == value && temp->marked == true) {
				return false;
			}
			else {
				temp = temp->next;


			}

		}
		return false;
	}



};