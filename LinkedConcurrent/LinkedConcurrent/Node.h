#pragma once
#include <cstddef>

template <class T>
class Node {
public:
	T data;
	int key;
	class Node* next;
	Node(T myData) {
		//settind data and hash key.  Next remains NULL
		data = myData;
		key = std::hash<std::T>()(data);
		next = NULL;
	}
};