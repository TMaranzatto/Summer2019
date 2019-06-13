
#include "Window.h"
#include <Limits.h>
#include <cstddef>


class NonBlockingLinkedList {
private:
	node* head;
public:
	NonBlockingLinkedList() {
		//initializing marks to be false
		//this could cause some errors later on, check this first for bugs
		node* head = new node(NULL, INT_MIN);
		node* nxt = new node(NULL, INT_MAX);
		head->next = nxt;
	}

	__global__
	bool add(int item) {
		//remember that the item is an int, and so is it's own hash
		while (true) {
			Window* window;
			node* pred;
			node* curr;

			cudaMallocManaged(&window, sizeof(window));
			cudaMallocManaged(&pred, sizeof(pred));
			cudaMallocManaged(&curr, sizeof(curr));

			window = find(head, item);
			pred = window->pred;
			curr = window->curr;

			if (curr->keyvalue == item) {
				cudaFree(window);
				cudaFree(pred);
				cudaFree(curr);
				return false;
			}

			else {
				//may cause race condition, not sure..
				//trying to make things work in our data struct
				curr->mark = false;

				node* nde;
				cudaMallocManaged(&nde, sizeof(nde));

				nde = new node(curr, item);

				//make this if condition a CAS statement for pred -> next
				if (true) {
					cudaFree(window);
					cudaFree(pred);
					cudaFree(curr);
					cudaFree(nde);

					return true;
				}
			}
		}

	}
	//NEED TO MAKE THIS ATOMIC!!!!!
	//MASSIVE RACE CONDITIONS HERE
	//cuda atomicExch() will be needed here

	__global__
	bool atomicAttemptMark(node* toTry, int expectedReference, bool newMark) {
		if (toTry->keyvalue == expectedReference){
			toTry->mark = &newMark;
			return true;
		}
		else {
			return false;
		}
	}

	__global__
	bool remove(int item) {
		bool snip;
		while (true) {

			Window* window;
			node* pred;
			node* curr;

			cudaMallocManaged(&window, sizeof(window));
			cudaMallocManaged(&pred, sizeof(pred));
			cudaMallocManaged(&curr, sizeof(curr));

			window = find(head, item);
			pred = window->pred;
			curr = window->curr;

			if (curr->keyvalue != item) {
				cudaFree(window);
				cudaFree(pred);
				cudaFree(curr);
				return false;
			}
			else {
				node* succ;
				cudaMallocManaged(&succ, sizeof(succ));
				succ = curr->next;

				snip = atomicAttemptMark(curr->next, succ->keyvalue, true);
				if (!snip) {
					continue;
				}
				//insert CAS operation here
				cudaFree(window);
				cudaFree(pred);
				cudaFree(curr);
				cudaFree(succ);
				return true;
			}

		}
	}

	__global__
	bool contains(int item) {
		bool marked[1] = { false };
		node* curr = head;
		while (curr->keyvalue < item) {
			curr = curr->next;
			//this needs to be atomic
			//could have nasty race conditions here
			node* succ = curr->next;
			marked[0] = succ->mark;

		}
		return(curr->keyvalue == item && !marked[0]);
	}
};