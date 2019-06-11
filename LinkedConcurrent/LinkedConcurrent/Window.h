#pragma once
#include "Node.h"
#include <cstddef>

class Window {
	class Node* pred;
	class Node* curr;
	Window(class Node* myPred, class Node* myCurr) {
		pred = myPred; curr = myCurr;
	}
};

Window find(class Node* head, int key) {
	class Node* pred = NULL;
	class Node* curr = NULL;
	class Node* succ = NULL;
	bool marked[] = { false };
	bool snip;

	while (true) {
		pred = head;
		curr = pred.next.
	}
}