#include <iostream>
#include "MegaAlloc.h"
#include <conio.h> //for getch()
#include <string>
using namespace std; 


int main() {

	MegaAlloc<int> s; 

	//s.operator new(sizeof(4));
	//ss.operator new(sizeof(5));
	//cout << s.getchunk();
	int a = 19;
	

	void * fds = s.bitallocate(sizeof(a));
	//s.allocate();
	//cout << "here we are";
	cout << endl; 
	cout << endl; 
	cout << "PRINT"<< endl;
	s.print();
	cout << endl; 
	cout << s.getchunk() << endl;
	//cout << s.getarenas();
	cout << endl; 
	cout << endl; 
	cout << endl; 

	//cout << "Finished";
	cout << fds; 

	
	//cout << endl; 
	//cout << whynot; 




	return 0; 
	
	
	


	//return 0; 
}