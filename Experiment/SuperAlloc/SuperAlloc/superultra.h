//Created by Nick Stone
#pragma once
#include <iostream>
#include <cstddef>
#include <stdlib.h>
#include <atomic>
#include <cstdint>
#include <bitset>
#include <thread>
#include <mutex>
using namespace std;



template <class T>

class superultra {

public:
	//Using mutex locks in order to start with the easiest concurrency

	mutex mut;

	//Template variables Starting with the values that are supposed to be held by the class
	using value_type = T;
	using pointer = T *;
	using const_pointer = const T*;
	//These variables will be used for allocating the memory IE void * commonly used
	using size_type = size_t;
	using void_star = void*;



private:
	//The number of allocations that occur in a given 
	int count = 0;
	//Each of these structs is used to form a simple linked list data structure
	//Each Arena is a given size of memory and points to the next arena in the list
	//Each arena also holds a size of the arena since this is dynamic as well as the location of the arena
	//Each arena is also capable of holding up to a certain number of 64 bit maps.. This detail can easily be changed
	//However the complications of bitset * and using 64 bit pointers makes freeing memory Non-Trivial
	struct Arena {
		//Where the next arena is stored in the list 
		struct Arena* next;
		//The size of the arena... 64 128 256 ....
		size_type Arenasize;
		//Location of where the arena starts this is used for allocating and sectioning off memory 
		void_star startarena;
		//Not used yet but will be for accessing nodes... Concurrency
		atomic_flag lock;
		//Needs to hold a number of bitsets ptrs equal to the bytes that each region holds 
		//For only 64 bytes
		pointer bitset; 
	};
	//For the overall program
	void_star start;
	//size_type nextArenaBlock;
	//static size_type numallocs;
	//Number of arenas in the list how many nodes are here and active
	int numarenas;
	//The size of the next block of memory ready to be allocated
	size_type chunk;
	//MALLOC
	//WHat is the total size we have allocated
	size_type HeapSize;
	//Size of the region we are going to allocate next
	size_type nexts;
	//MALLOC
	//Where the start of the arena is goingto be
	Arena* Head_Arena;
	//Location of the next node that needs to be created
	Arena* Next_Arena;
	//For the self referential Bit map 

public:
	//THis is apart of the template easier to assign aa then arena * 
	using aa = Arena *;

	//These are a series of quick function calles mainly called getters or accessors
	//These were used to make sure the proper data was being collected and used
	int getnumallocs() { return count; }
	int getnumarenas() { return numarenas; }
	int getarenas() { return numarenas; }
	size_type getchunk() { return chunk; }
	void_star getstart() { return start; }
	Arena* gethead() { return Head_Arena; }
	//End QUick Functions

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
		//Add the chunk to the overall size of the heap 
		HeapSize += chunk;

		//Increase the number of times that we have malloced
		count = count + 1;
		//Set the chunk size to be double the last allocation
		chunk = chunk * 2;
		//Return the next location that we have allocated... Points to an arena!
		return retval;
	}

	//This functions goal is to be used to manipulate a void pointer to then be returned to the programmer 
	//The function will take in the size_t of the position and then it will take the size of the object that needs
	//to be allocated then it will take in the start of the arena that the object is being allocated too
	//Then a new void pointer will be created and returned that will be pointing to an object within the 
	//Range of bits in the given arena
	void_star lowlevelalloc(size_type posstart, size_type thisbig, void_star arenast) {
		//Make a new void * pointer that will point to the arena start + the positition of the next free spot in the bit map 
		// + the overall size that needs to be allocated. 
		void_star newplace = &arenast + posstart + thisbig;
		//Return new value 
		return newplace;

	}

	//We take in the bit map from an arena and the size of the object that is trying to be allocated to the given region
	//we turn the bit map into a binary string and then check that string to make sure that it has a given number of 0s
	//in a row that will allow for an object to be allocated
	// If it has room we return true
	//if not enough room to allocate then we return false 
	//////////////////////////////////////////////////////////////////
	//Create the size we need
		//check to see if any map has that many 0s
	////////////////////////////////////////////////////////////////////
	//Has Room
	////////////////////////////////////////////////////////////////////

	bool checkroom(Arena * temp, size_type big) {
		//Set a flag and a counter to be false and 0
		bool flag = false;
		size_type counter = 0;
		//A double for loop in order to iterate through the bit map 
		//We know the given bitmap size is 64
		for (int i = 0; i <= temp->Arenasize; i++) {

			//This inner loop is to check that there are enough positions in the list in order to allocate 
			for (int j = i; j <= temp-> Arenasize; j++) {

				//If the count is equal to the size of the thing being allocated we have enough room and we can exit the program 
				if (counter == big) {

					flag = true;
					return flag;
				}
				//New if chain we check the next spot in the array if its a 0 increment the counter
				if (temp->bitset[j] == 0)
				{
					//Increment counter by 1
					counter += 1;

				}
				else {
					//Reset the counter to 0
					counter = 0;
				}

			}

			counter = 0;

		}
		cout << "NO ROOM ";
		//Traverse whole list and found nothing... Return false NO room to allocate
		return flag;


	}

	////////////////////////////////////////////////////////////////////


	//Swap 0s to 1s
	//////////////////////////////////////////////////////////////////////

	//This is designed to test a given map that belongs in a given arena
	//However we know that some arenas of smaller size may only need 1 map to represent them 
	//So we then take in a size of the object that needs to be allocated
	//Then we create a string that contains the number os 0s needed to allocate
	//Then if it has enough room then we update the string and send it back 
	//And we return the position in the string that has the 0s
	//If not then we return -1 meaning no room to allocate

	size_type changeset(size_type needbig, Arena* e) {
	
		size_type counter = 0;
		size_type positionchanging = 0;

	
		bool flag = false;
		bool flag2 = false;
		//int counter = 0; 
		 //cout << endl; 
		// cout << temporary << "TEMP HERE" ; 
		// cout << endl;
		while (flag == false) {
			for (int i = 0; i <= e->Arenasize; i++) {
				positionchanging = i;
				for (int j = positionchanging; j <= e->Arenasize; j++) {
					//cout << temporary[j] << endl;
					if (counter == needbig) {

						flag = true;
						flag2 = true;
						//i = 64; 
						break;
					}
					else if (e->bitset[j] == 0)
					{
						//Increment counter by 1
						counter += 1;
						//cout << counter;
					}
					else {
						//Reset the counter to 0
						counter = 0;
						break;
					}
				}
				if (flag2 == true) {
					break;
				}


				counter = 0;

			}
			cout << endl;
			//cout << positionchanging << " positionchaning" << endl;

			//cout << "SHIP -1";
			//return positionchanging; 
		}
		if (flag2 == false) {

			positionchanging = -1;

		}
		//scout << "WE GET HERE";
		int space = positionchanging;
		for (int i = 0; i < needbig; i++) {

			e->bitset[space] = 1;
			space = space + 1;
		}
		//cout << temporary;
		//s = s.replace(pos, needbig, nhold);
		//Send String to be uint
				//Set map to be new uint
		return positionchanging;


		//int pos = a.find(hold);
		//cout << hold; 
	}

	//////////////////////////////////////////////////////////////////////

	//This hub function serves as the main function that controls the overall allocation process that is used by this program 
	// We first check the arena size then based on the arena size we know how many maps need to be checked to see if therre is proper room
	//To be allocated... If there is no room to be allocated we return null. 

	void_star hub(Arena* e, size_type needbig) {


		if (checkroom(e, needbig) == true) {
			size_type s = changeset(needbig, e);
			return lowlevelalloc(s, needbig, e->startarena);
		}
		else {

			return NULL; 
		}
	}



	//Bitallocate -> individually allocates based on bitmap..
	//Set the memory to false and keep this loop going 
	//We then iterate through the linked list if therre is no room or any arenas in the linked list
	//Then we allocate a new arena and make sure the arena is in the linked list
	//We then make sure that the arena has enough room to allocate the given memory that has been input by the programmer

	void_star bitallocate(size_type needbig) {
		while (1) {
			if (mut.try_lock()) {
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
								mut.unlock();
								return store;
								//Success!

							}

						}
						if (memoryallocated == false) {
							allocate();
						}
					}
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
		cout << temp->Arenasize; 
		temp->startarena = temp;
		//temp->maps = new int[chunk /2]; 
		//temp->maps = int[Arenasize] f;
		//temp->maps[i] = 0;
		//int array[] = e->bitset;
		//int const n = sizeof(array) / sizeof(array[0]);
		temp->bitset = new int[temp ->Arenasize];
		//cout << "Still alive";
		// traverse through array and print each element
		for (int i = 0; i <= temp->Arenasize; ++i) {
			
			temp->bitset[i] = 0;
			//cout << temp->bitset[i];
		}
	


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
			start = Head_Arena;
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
			//cout << he << endl;

			Next_Arena->next = he;

		}

		//Function DOne
	}



	//First block is size 64 byte 
	//We start with no arenas and wait for the user to request memory before we alloate
	superultra()
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



	~superultra()
	{
		Arena* temp = Head_Arena;
		Arena * temps = temp; 
		if (Head_Arena != NULL) {
			delete(Head_Arena->bitset);
			if (Head_Arena->next != NULL) {
				temp = Head_Arena->next; 
				delete(Head_Arena);
				while (temp != NULL) {
					delete(temp->bitset);
					temps = temp; 
					temp = temp->next; 
					delete(temps);



				}


			}
			else {
				delete(Head_Arena);
			}




		}
	}


	// Frees a bit map in a given arenas
	void deallocate(void_star a) {

		free((Arena*) a);
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

			cout << "Map Looks like this";
			cout << endl;
			int *array = temp->bitset;
			//int const n = sizeof(array) / sizeof(array[0]);

			// traverse through array and print each element
			cout << endl; 
			for (int i = temp->Arenasize; i >= 0; --i) {
				cout << array[i];
			}

			//e->bitset = array;
			cout << endl;

			//cout << toBinary(temp->map) << " TO binary" << endl;
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
		
	

		// traverse through array and print each element
		for (int i = 0; i <= e->Arenasize; ++i) {
			e->bitset[i] =0;
		}
		 


	}
	//Function designed to take in a mega allocator and reconstruct the bit map 

	void recovery() {

		


		//Logic needs to be figured out after the allocator can allocate properly

	}



};


