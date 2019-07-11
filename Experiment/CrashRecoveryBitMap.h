#pragma once
#include <iostream>
#include <vector>

using namespace std; 




class CrashRecoveryBitMap {

private: 

	vector<bool> test;
	//Finished
	vector<bool> testing1;
	size_t totalSizeHeap; 
	//Finished
	int nextfits;
	void* start; 

	//Bit Map

public: 

	CrashRecoveryBitMap(size_t s, void* instart) {
		nextfits = 0; 
		totalSizeHeap = s; 
		for (int i = 0; i < 128; i++) {
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
	void recovery(vector<size_t> sg, vector<void*> ggg) {
		
		int offset = 128 * 1024; 

		for (int i = 0; i < ggg.size(); i++) {
			int temp = (int)ggg[i];
			int space = temp / offset; 
			test[space] = true; 
			cout << space << endl;


		}

		//int temp = (int)start; 
		//int n = ggg.size();
		//int help = (int)ggg[n-1];
		//int t = (int)ggg[1];

		//long letssee = help - temp;
		//cout << letssee; 
		//long ff = temp - t; 

		cout << endl; 
		//cout << ff; 

		for (int i = 0; i < sg.size(); i++) {

			totalSizeHeap += sg[i];

		}

	}


	void recover(void* testss, size_t s) {
		//Take in a Void * 
		test[s] = true; 
		

		//Take a Size 

		//First round is trivial if a Node exists in the next link in the chain from the List

		//Slap a 1 



	}

	vector<bool> realrecover(void * Arena, size_t arenaoffset,size_t arenasize) {
		void* temp = &Arena; 
		int loopit = (arenasize / arenaoffset);
		for (int i = 0; i < loopit; i++) {
			temp = &temp +  arenaoffset; 
			

			bool flag = (bool)temp; 
			if (temp == false) {
				testing1[i] = false;


			}
			else {


				testing1[i] = true;
			}



		}
		return testing1; 

		//Take in a size IE an arena of memory 

		//Loop through 


		//Find all allocated blocks 

		//Print 1 or 0 1 true - 0 false





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

	
	void print1() {

		int count = 0;
		for (int i = 0; i < testing1.size(); i++) {

			cout << testing1.at(i);
			cout << endl;
			count = i;

		}
		cout << endl << count;


	}









};