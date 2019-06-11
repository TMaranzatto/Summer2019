#pragma once
#include <cstddef>
#include <atomic>

struct node {
	node* next;
	int keyvalue;
	//key = std::hash<std::T>()(data);
};

class Window {
	node* pred;
	node* curr;
	Window(node* myPred, node* myCurr) {
		pred = myPred; curr = myCurr;
	}
};

Window find(node* head, int key) {
	node* pred = NULL;
	node* curr = NULL;
	node* succ = NULL;
	bool marked[] = { false };
	bool snip;
	while (true) {
		pred = head;
		curr = head -> next;
	}
}