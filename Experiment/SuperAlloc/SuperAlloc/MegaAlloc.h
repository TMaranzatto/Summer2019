#pragma once
#include <iostream>
#include <cstddef>
#include <list>
#include <stdlib.h>
//#include <vector>
#include <atomic>

using namespace std;



template <class T> 

class MegaAlloc {

public: 
	using value_type = T;
	using pointer = T *;
	using const_pointer = const T*;
	using size_type = size_t;
	using void_star = void*;
	using bitmap = int[5];
	

private:

	struct Arena {
		struct Arena* next;
		int maps[5];
		size_type Arenasize;
		void_star startarena;
		atomic_flag lock;
		//int nextfits;

	};
	//For the overall program
	void_star start;
	//size_type nextArenaBlock;
	//static size_type numallocs;
	int numarenas;
	size_type chunk;
	//MALLOC
	size_type HeapSize;
	size_type nexts;
	//MALLOC


	Arena* Head_Arena;
	Arena* Next_Arena;


	//For the self referential Bit map 




public: 
	using aa = Arena *;




	size_type getchunk() {

		return chunk;
	}

	//Malloc -> Moves the program counter 

	void_star malloc() {
		//size_type pad_size = chunk;
		void_star retval = &start + chunk;
		//nexts += pad_size;
		HeapSize += chunk;
		//numallocs += 1;
		chunk = chunk * 2;
		return retval;


	}

	//Bitallocate -> individually allocates based on bitmap..
	void_star bitallocate() {

	}

	Arena* arenainfo(Arena* temp) {
		//Arenasize  =chunk /2
		//ArenaStart = temp
		temp->Arenasize = chunk / 2;
		temp->startarena = temp;
	
		for (int i = 0; i < 4; i++) {
			temp->maps[i] = 0;

		}
		
		//temp->maps = test;

		return temp;
	}

	void allocate() {
		//numallocations += 1; 
		numarenas = numarenas + 1;
		Arena* e;

		//0 Elements;
		if (Head_Arena == NULL) {

			e = reinterpret_cast<Arena*>(malloc());
			//Set all of Node values in e; 
			//SEt head Node to E; 

			e = arenainfo(e);
			//SEt Tail Not to HeadNode -> next
			Head_Arena = e;
			/*
			Call Functions to Establish Node
			*/

			//Head Node Now E
			Next_Arena = Head_Arena->next;



		}
		else {
			/*
			Make New Node

			*/
			e = reinterpret_cast<Arena*>(malloc());

			e = arenainfo(e);


			Next_Arena = e;
			Next_Arena = Next_Arena->next;

		}
	}


	MegaAlloc()
	{
		numarenas = 1; 
		HeapSize = 64; 
		chunk = 64; 
		Head_Arena = NULL; 
		Next_Arena = NULL; 
		allocate();
		start = Head_Arena->startarena; 
	
		//Call Allocate 


	}

	~MegaAlloc()
	{



	}

	//Allocate -> Calls Malloc and Sets a new Arena


	void deallocate(void_star e, size_type s) {

		reinterpret_cast<Arena*>(e)->next = Next_Arena;
		Next_Arena = reinterpret_cast<Arena*>(e);


	}




	/*
	Allocator
	Malloc
	Free
	Traverse
	Arena
	*/


	/*
	BitMap
	
	*/
	//Returns the position in the bit array that is next to be allocated
	
	/*int nextfit(arena * e ) {
		
		bitmap test = e->maps; 
		int nextfits = 0;
		int temp = e-> nextfits;
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
	
	int firstfit(arena* e) {
		bitmap test = e->maps; 
		for (int i = 0; i < test.size(); i++) {
			if (test[i] == false) {
				return i;
			}
		}
	}
	//Return the range of positions that are needed and free to allocate the given size too
	int findpos(int num, arena * e)
	{
		bitmap test = e->maps; 
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
	void flipbits(int pos, arena* e) {
		bitmap test = e-> maps;
		bool t = test.at(pos);
		if (t == false) {
			t = true;
			test[pos] = t;
		}
		else {
			t = false;
			test[pos] = t;
		}
		e->map = test; 

	}


	//Kill and Free an Entire Arena 
	void killmap(arena* ar) {
		bitmap temp = ar->maps; 
		for (int i = 0; i < temp.size(); i++) {
			
			temp[i] = false; 



		}
		ar->maps = temp; 

		

	}

	*/

	//





	//End bit Maps



	//Static Calls...
/*
	static void_star operator new(size_type s) {
		return allocate(s);

	}
	static void operator delete(void_star p, size_type s) {

		deallocate(p, s);
	}
	*/


	//Performance 
/*
	size_type get_allocations() const
	{
		return numallocs;
	}
	*/

	










};


