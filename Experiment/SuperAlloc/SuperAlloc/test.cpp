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
	string hi = "hello world";
	int test = hi.find(" ");
	cout << test; 

	size_t needbig = 4; 
	string hold = "";
	string whynot = "100001";
	string nhold = "";
	for (int i = 0; i < needbig; i++) {

		hold = hold + "0";
		nhold = nhold + "1";
	}

	cout << hold;

	int pos = whynot.find(hold);
	whynot.substr();
	cout << endl; 
	cout << endl; 
	cout << endl; 
	cout << pos; 







	return 0; 
	
	
	


	//return 0; 
}