#pragma once
#include <cinttypes>
#include "Simpleallocator.h"



struct foo {
	uint64_t a[2];

	static Simpleallocator alloc; 
	static void* operator new(size_t size) {
		return alloc.allocate(size);

	}
	static void operator delete(void* p, size_t s) {

		alloc.deallocate(p, s);
	}




};



Simpleallocator foo::alloc; 







