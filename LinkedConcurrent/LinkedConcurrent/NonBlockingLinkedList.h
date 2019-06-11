#pragma once
#include "Window.h"
#include "Node.h"
#include <Limits.h>

template <class T>
class NonBlockingLinkedList {
private:
	class Node* head;
public:
	NonBlockingLinkedList() {
		head = new Node(INT_MIN);
		head.next = new Node(INT_MAX)
	}

};