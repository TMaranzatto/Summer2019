#pragma once
#include <iostream>
#include <stdlib.h>
#include <queue>
#include <sys/mman.h>


using namespace std;


class mapalloc {

private:
	struct allocator {
		void* ad; 
		size_t len; 
		int pro; 
		int flags; 
		int fd; 
		off_t offset;
		
		
		
		
		void* base;
		size_t regionsize;
		size_t next;
		int blocks;
		priority_queue<void*> freed;
		priority_queue<void*> alloc;
		//free[];
   //	 allocated[];


		void deal(void* n) {

			free(n);
			freed.push(n);
			blocks = blocks - 1;

		}


		//Takes in all of the Parameters for a file to be Memory Mapped

		allocator(void* aadr, size_t ilength, int iprot, int iflags, int ifd, off_t ioffset) {
			ad = aadr; 
			len = ilength;
			pro = iprot; 
			fd = ifd; 
			offset = ioffset; 

			



			//regionsize = heap_size;
			//next = 0;
			//blocks = 0;
			//base = malloc(heap_size);



		}
		void map() {

		}
		void unmap() {

		}

		void* alloc(size_t s) {
			if (freed.size() == 0) {
				size_t temp = s;
				size_t pad_size = PAD(temp);
				void* retval = &base + next;
				next += pad_size;
				alloc.push(retval);
				return retval;
			}
			else {
				void* hold = freed.top();
				freed.pop();
				return hold;
				//return freed.pop();
			}
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

