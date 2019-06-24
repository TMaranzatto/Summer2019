#pragma once
#include <iostream>
#include <stdlib.h>

using namespace std; 


class aalloc {

private: 
	struct allocator {
		void* base;
		size_t regionsize; 
		size_t next; 

		void free(void* n) {

			free(n);
		}

		allocator(size_t heap_size) {

		}

		void*malloc(size_t s) {
			size_t pad_size = PAD(size);
			void* retval = regionsize + next;
			next += pad_size;
			return retval; 

		}

		size_t PAD(size_t s) {
			size_t temp = s; 
			if (s % 8 == 0) {
				return temp; 

			}
			else {
				temp = s / 8;
				size_t offset = 8 - temp;
				temp = s + offset;
				return temp; 

			}

			//return temp; 
		}





	};





public: 





};

