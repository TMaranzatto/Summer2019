#include <iostream>

#include "foo.h"


using namespace std; 


int main() {

	
	int arraysize = 10; 
	foo* f[10];
	cout << "size of(foo)" << sizeof(foo) << endl; 
	for (int i = 0; i < arraysize; i++) {

		f[i] = new foo();
		cout << f[i] << endl; 
	}
	cout << "Delete now";
	for (int i = 0; i < arraysize; i++) {
		delete f[i];
		cout << "Delete" << endl;
		cout << i; 

	}
	cout << "Finish Delete";

	return 0; 



}