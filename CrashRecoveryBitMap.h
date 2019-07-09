#pragma once
#include <iostream>
#include <vector>

using namespace std; 




class CrashRecoveryBitMap {

private: 

	vector<bool> test;
	int nextfits;
	void* start; 

	//Bit Map

public: 

	CrashRecoveryBitMap(size_t s, void* instart) {
		nextfits = 0; 
		for (int i = 0; i < s; i++) {
			test.push_back(false);



		}
		start = instart; 
		//Instantiate Bit map 

	}

	~CrashRecoveryBitMap() {
		
		
		//delete(test);
		//Idk 


	}
	//Take something in and rebuild the bit map
	void recover(void* testss, size_t s) {
		//Take in a Void * 
		test[s] = true; 
		

		//Take a Size 

		//First round is trivial if a Node exists in the next link in the chain from the List

		//Slap a 1 



	}


	//Returns the position in the bit array that is next to be allocated
	int nextfit() {
		int temp = nextfits; 
		if (nextfits == false) {
			for (int i = nextfits; i < test.size(); i++) {
				if (test[i] == false) {
					nextfits = i; 
					break;
				}

			}

			return temp; 

		}
		for (int i = nextfits; i < test.size(); i++) {
			if (test[i] == false) {
				return i; 


			}



		}


	}


	int firstfit() {
		for (int i = 0; i < test.size(); i++) {

			if (test[i] == false) {
				return i; 

			}

		}

	}

	//Return the range of positions that are needed and free to allocate the given size too
	int findpos(int num ) 
	{
		int start = 0; 
		int temp = 0; 
		for (int i = 0; i < test.size(); i++) {
			temp = 0; 
			if (test[i] == false) {
				start = i; 
				
				for (int j = i; j < test.size(); j++) {
					if (temp == num) {
						return start; 
						//break; 
					}
					else if (test[j] == false) {
						temp = temp + 1; 
					}
					else {

						break;
					}
					
				}
				continue; 
				


			}


		}

		cout << "not enough room to allocate";

	}

	//Flip The bit at a given position in the array
	void flipbits(int pos) {

		bool t = test.at(pos);
		if (t == false) {

			t = true; 
			test[pos] = t;

		}
		else {
			t = false; 
			test[pos] = t; 

		}

	}


	//Print all of the contents of the vector
	void print() {

		int count = 0; 
		for (int i = 0; i < test.size(); i++) {

			cout << test.at(i);
			cout << endl; 
			count = i; 

		}
		cout << endl << count;


	}

	/*
		What should this program do? 
		Functions: 

		Recover the Data 

		Namely Build or Rebuild the bit map according the linked list that is taken in 

		Return the Next Fit Position in the array 

		Update the array When the Linked list is updated
		 	
	*/










};