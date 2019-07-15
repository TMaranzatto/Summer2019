#pragma once
#include <iostream>
#include <cstddef>
#include <list>
#include <stdlib.h>


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

	//Arena Allocator
	void_star allocate(size_type s) {

		freenode* e;
		if (!next_store) {
			size_t chunk = num_chunk + s;
			next_store = e = reinterpret_cast<freenode*> (malloc(chunk));
			for (int i = 0; i < num_chunk + 1; i++)
			{
				e->next = reinterpret_cast<freenode*>(reinterpret_cast<char*>(e) + s);
				e = e->next;


			}
			e->next = nullptr;


		}
		e = next_store;
		next_store = next_store->next;
		return e;






	}
	void deallocate(void_star p, size_type s) {

		reinterpret_cast<freenode*>(e)->next = next_store;
		next_store = reinterpret_cast<freenode*>(e);


	}

	static void_star operator new(size_type s) {
		return allocate(s);

	}
	static void operator delete(void_star p, size_type s) {

		deallocate(p, s);
	}

	//END Arena Allocator

	//Begin Bit Map


	//End Bitmap 






	//Begin everything else 


	//Return the number of times things have been allocated... This is purely for performance
	size_type get_allocations() const
	{
		return mAllocations;
	}



private:
	//Private fields

	//Allocation
	static size_type numallocations;
	struct freenode {

		struct freenode* next;

	};
	freenode* next_store = nullptr;
	static const size_type num_chunk = 4;
	//End Allocation

	//Begin Bitmap


	//End Bitmap

};