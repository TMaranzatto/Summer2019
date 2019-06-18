#pragma once
#include <iostream>
using namespace std;

class Heap {
private: 

	int size; 
	int* array; 
	int position;

	//vector <int> heaps; 



public: 
	Heap() {
		size = 5;
		position = 0; 
		array = new int[size];
		
	}
	~Heap() {
		delete[]array;
	}

	void print() {

		for (int i = 0; i < size; i++) {
			cout << array[i] << "           ";

		}

	}
	
	void insert(int val) {
		if (position == size) {
			resize();
		}
		array[position] = val; 
		position = position + 1; 
		

	}

	void remove() {
		if (position == 0) {
			cout << "List is empty";
		}
		else {
			array[position-1] = NULL;
			position = position - 1;
		}
	}
	void resize() {
		int* temp = new int[size * 2];
		//size = size * 2; 
		for (int i = 0; i < size; i++) {
			temp[i] = array[i];


		}
		size = size * 2; 
		array = temp; 


	}
	bool contains(int val) {
		for (int i = 0; i < size; i++) {
			if (array[i] == val) {
				cout << "FOund at position" << i << endl; 
				return true; 
			}

		}
		cout << "NOT FOUND" << endl; 
		return false; 

	}


};