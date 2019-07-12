#pragma once
#include <iostream>
#include <vector>



class bitmaps {
private: 


	size_t totalSizeHeap;
	//Finished
	int nextfits;
	void* start;
	vector<bool> test; 





public: 
	bitmaps() {



	}

	~bitmaps() {
		//Nothing yet

	}


	void recover() {




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
	int findpos(int num)
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





















};










