#pragma once
#include <iostream>
#include <cstddef>
#include <stdlib.h>
#include <atomic>
//Need to include Uinptrs
#include <cstdint>

using namespace std;



template <class T> 

class MegaAlloc {

public: 
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
		atomic_uint64_t map;
		//for 128 Bytes
		atomic_uint64_t map1;
		
		atomic_uint64_t map2;
		atomic_uint64_t map3;
		atomic_uint64_t map4;
		atomic_uint64_t map5;
		atomic_uint64_t map6;
		atomic_uint64_t map7;
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


	int getarenas() {return numarenas; 	}

	size_type getchunk() {	return chunk;}

	//Malloc -> Moves the program counter 
	void_star malloc() {
		void_star retval = &start + chunk;
		HeapSize += chunk;
		count = count + 1;
		chunk = chunk * 2;
		return retval;
	}

	void_star lowlevelalloc(size_type posstart, size_type big, void_star arenast) {

		void_star newplace= &arenast + posstart;
		newplace = newplace + big; 
		return newplace; 

	}

	findholes(size_type needbig , atomic_uint64_t a ) {
		atomic_uint64_t temporary = a;





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
					if (temp->Arenasize == 64) {
						//Search the map
						//If found allocate
						//If not temp = temp -> next

					}
					else if (temp->Arenasize == 128) {

					}
					else if (temp->Arenasize == 256) {

					}
					else if (temp->Arenasize == 512) {


					}
					else {

						//1024 Search all 

					}

				}
				if (memoryallocated == false{
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
		//ArenaStart = temp
		temp->Arenasize = chunk / 2;
		temp->startarena = temp;
		//temp->maps = new int[chunk /2]; 
		//temp->maps = int[Arenasize] f;
		//temp->maps[i] = 0;
		temp->map = 0; 
		temp->map1 = 0; 
		temp->map2 = 0;
		temp->map3 = 0; 
		temp->map4 = 0; 
		temp->map5 = 0; 
		temp->map6 = 0; 
		temp->map7 = 0; 
		
		
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
			/*
			Call Functions to Establish Node
			*/
			//Head Node Now E
			Next_Arena = Head_Arena->next;
		}
		else if (Head_Arena->next == NULL) {

			Arena* te;
			te = reinterpret_cast<Arena*>(malloc());
			te = arenainfo(te);
			Head_Arena->next = te; 
			
		}
		else {
			//Arena* temp; 
			Arena* he;
			Next_Arena = Head_Arena; 
			while (Next_Arena -> next != NULL) {

				//Get to the last Node
				Next_Arena = Next_Arena->next; 

			}

			
			/*
			Make New Node

			*/
			he = reinterpret_cast<Arena*>(malloc());

			he = arenainfo(he);
			cout << he << endl;
			
			Next_Arena -> next = he;
			//Next_Arena->next = NULL;
			//Next_Arena = Next_Arena->next;
		}

		//Function DOne
	}


	MegaAlloc()
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

	~MegaAlloc()
	{



	}

	//Allocate -> Calls Malloc and Sets a new Arena


	void deallocate(void_star e, size_type s) {

		reinterpret_cast<Arena*>(e)->next = Next_Arena;
		Next_Arena = reinterpret_cast<Arena*>(e);
	}
	void print() {
		Arena* temp; 
		temp = Head_Arena;
		int count = 0; 
		while (temp != NULL) {

			//cout << temp->startarena; 
			
			cout << temp->startarena << " WIth the position in the linked list as " << count << endl; 
			count = count + 1; 
			cout << temp->map << endl;
			cout << toBinary(temp->map);
			//cout << count << endl; 
			//cout << endl; 
			//cout << temp->maps[0];
			temp = temp->next; 
		}
		cout << "Finished";



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
	
	//Bit map functions
	string toBinary(const T& t)
	{
		string s = "";
		int n = sizeof(T) * 8;
		for (int i = n; i >= 0; i--)
		{
			s += (t & (1 << i)) ? "1" : "0";
		}
		return s;
	}







};


