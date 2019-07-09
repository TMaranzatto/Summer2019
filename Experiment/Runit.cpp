#include <iostream>
#include "CrashRecoveryBitMap.h"


using namespace std; 



int main() {


	size_t t = 8;
	void* tt = malloc(sizeof(t)); 
	void* ttt = malloc(t);
	CrashRecoveryBitMap m(t,tt); 
	cout << endl; 

	cout << tt; 
	cout << endl; 
	cout << ttt; 
	
	//void *  b = ttt - tt; 

	//int g = static_cast<int>(tt); 

	int g = (int)tt; 
	cout << endl; 
	cout << g; 
	void* gg = (void*)g; 
	cout << endl; 
	cout << gg; 
	cout << endl; 
	cout << "-----------";
	m.print();
	m.flipbits(3);
	cout << endl; 
	m.print();
	//out << "Hello Universe";





	return 0;
};