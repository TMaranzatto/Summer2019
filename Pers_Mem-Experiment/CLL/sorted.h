#pragma once
#include <iostream>
using namespace std; 


class sorted {
private: 
	int size
	vector<int> heap;
	vector<int> temp; 


public: 

	sorted() {

		heap.insert(0);
		temp.insert(0);

	}
	
	~sorted() {
		if (heap) {
			delete(heap);
		}
	}


	void insert(int val) {
 
		heap.push_back(val);
		sort();
	}
	void remove() {

		heap.pop_back();
	}
	bool search(int val) {
		for (int i = 0; i < heap.size(); i++) {
			if (heap[i] == val) {
				cout << "Found" << endl; 
				return true; 
			}


		}
		cout << "Not FOund";
		return false; 

	}
	void print() {
		for (int i = 0; i < heap.size(); i++) {
			cout << heap[i] ;

		}

	}
	void sort() {
		
		for (int i = 0; i < heap.size(); i++) {
			int min = heap[0];
			for (int j = 0; j < i; j++) {
				if (heap[j] < min) {
					min = heap[j];
				}
				


			}
			temp.insert(min);
			heap.remove(heap[j])
		}
		heap = temp; 

	}
};