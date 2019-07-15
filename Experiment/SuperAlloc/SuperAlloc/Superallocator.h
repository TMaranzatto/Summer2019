#pragma once
#include <iostream>
//#include <cstddef>
#include <list>
#include <stdlib.h>
#include <vector>


using namespace std; 
/*
	Created by Nick Stone
	Created on 7/11/2019
	Super allocator
	This allocator is a pool allocator that uses bit maps in order to maintain and dispense memory locations
	Further additions will be added
*/

template <class T>
class Superallocator {

public:

	//Template stuff
	using value_type = T;
	using pointer = T *;
	using const_pointer = const T*;
	using size_type = size_t;
	using void_star = void*;
	using bitmap = vector<bool>; 

	//Constructor

	Superallocator() {

		//Come back later

	}
	//Destructor
	~Superallocator() {
		if (next_Arena) {
			Arena* lead = next_Arena->next;
			Arena* follow = next_Arena;
			while (lead) {
				delete(follow);
				follow = lead;
				lead = lead->next;
			}
			delete(follow);
		}


	}


	//Arena Allocator



	void_star allocate(size_type s) {
		//numallocations += 1; 
		Arena* e;
		if (!next_Arena) {
			size_t chunk = num_chunk + s;
			next_Arena = e = reinterpret_cast<Arena*> (malloc(chunk));
			for (int i = 0; i < num_chunk+1; i++)
			{
				e->next = reinterpret_cast<Arena*>(reinterpret_cast<char*>(e) + s);
				e = e->next;


			}
			e->next = nullptr;


		}
		e = next_Arena;
		next_Arena = next_Arena->next;
		//cout << "Allocation complete";
		return e;
	}
	void deallocate(void_star e, size_type s) {

		reinterpret_cast<Arena*>(e)->next = next_Arena;
		next_Arena = reinterpret_cast<Arena*>(e);


	}

	static void_star operator new(size_type s) {
		return allocate(s);

	}
	static void operator delete(void_star p, size_type s) {

		deallocate(p, s);
	}
	void_star malloc(size_type s) {

		size_type pad_size = PAD(s);        
		void_star retval = &start + nexts;
		nexts += pad_size;
		HeapSize += pad_size;
		return retval;


	}
	size_type PAD(size_type s) {
		size_type temp = s;
		if (s % 8 == 0) {
			return temp;

		}
		else {
			temp = s / 8;
			size_type offset = 8 - temp;
			temp = s + offset;
			return temp;

		}

		//return temp; 
	}

	//END Arena Allocator
	//Begin Bit Map
		//Returns the position in the bit array that is next to be allocated
	int nextfit(bitmap test) {
		int nextfits = 0;
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
		else {
			for (int i = nextfits; i < test.size(); i++) {
				if (test[i] == false) {
					return i;
				}
			}
		}
	}
	int firstfit(bitmap test) {
		for (int i = 0; i < test.size(); i++) {
			if (test[i] == false) {
				return i;
			}
		}
	}
	//Return the range of positions that are needed and free to allocate the given size too
	int findpos(int num, bitmap test)
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
	bitmap flipbits(int pos, vector<bool> te) {
		bitmap test = te;
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

	//End Bitmap
	//Begin everything else 
	//Return the number of times things have been allocated... This is purely for performance
	size_type get_allocations() const
	{
		return numallocations;
	}

	void print() {
		Arena* temp = next_Arena; 
		while (temp != nullptr) {

			cout << temp << " HERO" << endl;
			temp = temp->next; 
		}
		cout << "FINISHED" << endl;
	}

private:
	//Private fields

	//Allocation
	static size_type numallocations;
	struct Arena {

		struct Arena* next;

	};
	struct BM {
		bitmap bitmapdata; 
		size_type overallsize; 
		
		size_type offset; 
		void_star initial; 

		BM(size_type si) {
			this->overallsize = si; 

			for (int i = 0; i < overallsize; i++) {
				bitmapdata[i] = false;
			}
		}

	};
	BM* bitmaps = nullptr;
	Arena* next_Arena = nullptr;
	static const size_type num_chunk = 4;
	void_star start; 
	size_type HeapSize;
	size_type nexts; 
	



	//End Allocation

	//Begin Bitmap


	//End Bitmap

};


int main() {

	Superallocator<int> s;

	//s.operator new(sizeof(4));
	//ss.operator new(sizeof(5));
	s.allocate(sizeof(4));
	s.allocate(sizeof(5));
	s.print();

	cout << "Mark Reached";

	//Exited with a code 3




	return 0; 
}