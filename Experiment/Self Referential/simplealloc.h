#pragma once
#include "Simpleallocator.h"
#include <stdlib.h>


void* Simpleallocator::allocate(size_t s) {
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


void Simpleallocator::deallocate(void* e, size_t) {

	reinterpret_cast<freenode*>(e)->next = next_store; 
	next_store = reinterpret_cast<freenode*>(e);


}







