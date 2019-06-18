#pragma once
#include <iostream>
 

using namespace std; 


class sort {
private: 

	int size;
	int elements;
	int* array;
	//int position;




public: 

	sort() {
		size = 5;
		elements = 0; 
		//position = 0;
		array = new int[size];

	}
	~sort() {
		delete[]array;
	}


	void insert(int val) {
		//int min = array[0];
		if (elements == size || elements == size-1) {
			resize();
		}
		array[size - 1] = val; 
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < i; j++) {
				if (array[i] > array[j]) {
					int temp = array[i];
					array[i] = array[j];
					array[j] = temp;
					//break; 

				}
			}

		}
		elements = elements + 1; 
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
	void sorts() {

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < i; j++) {
				if (array[i] > array[j]) {
					int temp = array[i];
					array[i] = array[j];
					array[j] = temp;
					//break; 

				}
			}
		}



	}
	void remove(int val) {

		for (int i = 0; i < size; i++) {
			if (array[i] == val) {
				array[i] = NULL;
				sorts();
			}

		}
		cout << val << "Not in list" << endl; 
		elements = elements - 1; 

	}

	bool contains(int vall ) {
		for (int i = 0; i < size; i++) {
			if (array[i] == vall) {
				cout << "FOund at position" << i << endl;
				return true;
			}

		}
		cout << "NOT FOUND" << endl;
		return false;
	}

	void print() {
		for (int i = 0; i < size; i++) {
			cout << array[i] << "       ";

		}
	}







};
