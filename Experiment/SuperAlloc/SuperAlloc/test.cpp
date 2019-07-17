#include <iostream>
#include "MegaAlloc.h"
#include <conio.h> //for getch()
using namespace std; 


int main() {

	MegaAlloc<int> s; 

	//s.operator new(sizeof(4));
	//ss.operator new(sizeof(5));
	//cout << s.getchunk();
	int a = 19;
	cout << "Allocated once";


	s.allocate();
	cout << "Allocated twixe";
	s.allocate();
	cout << "Allocated three";
	//s.allocate();
	//cout << "here we are";
	cout << endl; 
	cout << endl; 
	cout << "PRINT"<< endl;
	s.print();
	cout << endl; 
	cout << s.getchunk() << endl;
	cout << s.getarenas();
	cout << endl; 
	cout << endl; 
	cout << endl; 

	//cout << "Finished";


	return 0; 

	
	


	//return 0; 
}