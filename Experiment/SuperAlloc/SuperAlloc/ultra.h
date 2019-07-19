//Created by Nick Stone
#pragma once
#include <iostream>
#include <cstddef>
#include <stdlib.h>
#include <atomic>
#include <cstdint>
#include <bitset>


using namespace std;



template <class T>

class ultra {

public:
	//
	using value_type = T;
	using pointer = T *;
	using const_pointer = const T*;
	using size_type = size_t;
	using void_star = void*;



private:
	int count = 0;
	struct Arena {
		struct Arena* next;
		size_type Arenasize;
		void_star startarena;
		atomic_flag lock;
		//Needs to hold a number of U int ptrs equal to the bytes that each region holds 
		//For only 64 bytes
		void_star bits; 
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


	int getarenas() { return numarenas; }

	size_type getchunk() { return chunk; }
	void_star getstart() { return start; }


	//Malloc -> Moves the program counter 
	//The goal of the arena allocation is to actually allocate more memory as few as times as possible. 
	//Thus meaning that everytime we allocate we set our chunk size or the amount of memory that we are allocating
	//To be chunk * 2 or to appear in 64 bits then 128 bits then 256 bits then 512.... etc. 

	void_star malloc() {

		//Take in no parameters
		//Create a new void star that points to the first position and adds the size to the void pointer
		//THen we increment the global counters and chunk for the next allocation to happen
		//Return the new pointer that is pointing to the new arena
		void_star retval = &start + chunk;
		HeapSize += chunk;
		count = count + 1;
		chunk = chunk * 2;
		return retval;
	}

	//This functions goal is to be used to manipulate a void pointer to then be returned to the programmer 
	//The function will take in the size_t of the position and then it will take the size of the object that needs
	//to be allocated then it will take in the start of the arena that the object is being allocated too
	//Then a new void pointer will be created and returned that will be pointing to an object within the 
	//Range of bits in the given arena
	void_star lowlevelalloc(size_type posstart, size_type thisbig, void_star arenast) {

		void_star newplace = &arenast + posstart + thisbig;

		return newplace;

	}

	//We take in the bit map from an arena and the size of the object that is trying to be allocated to the given region
	//we turn the bit map into a binary string and then check that string to make sure that it has a given number of 0s
	//in a row that will allow for an object to be allocated
	// If it has room we return true
	//if not enough room to allocate then we return false 

	bool hasroom(uint64_t s, size_type big) {
		



	}


	size_type findholes(size_type needbig, Arena* e, int maps) {
		
		//int pos = a.find(hold);
		//cout << hold; 
	}


	void_star hub(Arena* e, size_type needbig) {





		
			if (hasroom(e->bit, needbig) == true) {
				size_type s = findholes(needbig, e, 0);
				return lowlevelalloc(s, needbig, e->startarena);

			}
			else {

				return NULL;
			}

		
		




	}



	//Bitallocate -> individually allocates based on bitmap..
	void_star bitallocate(size_type needbig) {
		bool memoryallocated = false;

		while (memoryallocated == false) {
			if (Head_Arena == NULL) {
				allocate();

			}
			else {
				//Traverse all arenas and find a suitable place
				Arena* temp = Head_Arena;
				while (temp != NULL) {
					void_star store = hub(temp, needbig);
					if (store == NULL) {
						//Failed
						temp = temp->next;

					}
					else {
						return store;
						//Success!

					}

				}
				if (memoryallocated == false) {
					allocate();
				}
			}
		}
		//Check Head Node... 
		//If no head Node allocate
		//Search the head nodes array... Lets fix that while we are here 
		//Then go to the next node.. If its NULL allocate
		//
	}

	Arena* arenainfo(Arena* temp) {
		//Arenasize  =chunk /2
		//startarena = temp
		temp->Arenasize = chunk / 2;
		temp->startarena = temp;
		//temp->maps = new int[chunk /2]; 
		//temp->maps = int[Arenasize] f;
		//temp->maps[i] = 0;
		bitset<chunk / 2> bitmap = 0; 
		bitset<chunk / 2> * bitmapptr = bitmap; 
		temp->bits = (void_star) bitmapptr; 


		//temp->maps = test;

		return temp;
	}

	void allocate() {
		//numallocations += 1; 
		numarenas = numarenas + 1;
		//0 Elements;
		if (Head_Arena == NULL) {
			Arena* e;
			e = reinterpret_cast<Arena*>(malloc());

			//Set all of Node values in e; 
			//SEt head Node to E; 
			e = arenainfo(e);
			//SEt Tail Not to HeadNode -> next
			Head_Arena = e;
			Head_Arena->next = NULL;
			start = Head_Arena->startarena;
			/*
			Call Functions to Establish Node
			*/
			//Head Node Now E
			Next_Arena = Head_Arena->next;
		}
		else if (Head_Arena->next == NULL) {

			Arena* te;
			te = reinterpret_cast<Arena*>(malloc());
			//te->startarena = te; 
			te = arenainfo(te);
			Head_Arena->next = te;

		}
		else {
			//Arena* temp; 
			Arena* he;
			Next_Arena = Head_Arena;
			while (Next_Arena->next != NULL) {

				//Get to the last Node
				Next_Arena = Next_Arena->next;

			}


			/*
			Make New Node

			*/
			he = reinterpret_cast<Arena*>(malloc());

			he = arenainfo(he);
			cout << he << endl;

			Next_Arena->next = he;
			//Next_Arena->next = NULL;
			//Next_Arena = Next_Arena->next;
		}

		//Function DOne
	}


	ultra()
	{
		numarenas = 0;
		HeapSize = 64;
		chunk = 64;
		Head_Arena = NULL;
		Next_Arena = NULL;

		//allocate();
		//start = Head_Arena->startarena; 
		//Call Allocate 
	}



	~ultra()
	{


	}


	// Frees a bit map in a given arenas
	void deallocate(Arena* a) {

		free(a);
	}


	//This function will iterate through the current linked list and will print the bit map in the arenas 
	//As well as some other information to ensure that the right information is being passed
	void print() {
		Arena* temp;
		temp = Head_Arena;
		int counter = 0;
		while (temp != NULL) {

			//cout << temp->startarena; 

			cout << temp->startarena << " WIth the position in the linked list as " << counter << endl;

			cout << endl;

			cout << endl;

			counter = counter + 1;
			//cout << temp->map << endl;

			cout << endl; 
			cout << temp->bits; 
			cout << endl; 

			//cout << count << endl; 
			//cout << endl; 
			//cout << temp->maps[0];
			temp = temp->next;
		}
		cout << "Finished";
	}

	//Set every value each 64 bit pointer to be 0
	//THis will allow the algorithm to revisit pointers
	//and reallocate all of the values
	void free(Arena* e) {
		e->bits = NULL;
	}







	//Function designed to take in a mega allocator and reconstruct the bit map 

	void recovery() {
		//Logic needs to be figured out after the allocator can allocate properly

	}



};


