#include <iostream>
#include "MegaAlloc.h"
#include <conio.h> //for getch()
#include <string>
#include <cmath>
#include <bitset>
using namespace std; 


int main() {

	MegaAlloc<int> s; 

	//s.operator new(sizeof(4));
	//ss.operator new(sizeof(5));
	//cout << s.getchunk();
	int a = 19;
	int b = 200; 
	int c = 300; 
	int d = 4;
	int strawman = 23;
	int strawnan1 = 344;
	int strawman2 = 495;
	int	strawman3 = 399; 
	int strawman4 = 495; 

	void* dptr = s.bitallocate(sizeof(d));
	void* straw = s.bitallocate(sizeof(strawman));
	void * straw1  = s.bitallocate(sizeof(strawnan1));	
	void * straw2 = s.bitallocate(sizeof(strawman3));
	void * straw3 = s.bitallocate(sizeof(strawman2));
	void * straw4 = s.bitallocate(sizeof(strawman4));


	void * fds = s.bitallocate(sizeof(a));
	//s.allocate();
	void* letssee = s.bitallocate(sizeof(b));
	
	//cout << "here we are";
	void* strt = s.getstart();

	size_t temp = (size_t)strt - (size_t)fds; 

	cout << endl; 
	cout << endl; 
	cout << "PRINT"<< endl;
	s.print();
	cout << endl; 
	//cout << s.getchunk() << endl;
	//cout << s.getarenas();
	cout << endl; 
	cout << endl; 
	cout << endl; 



	//cout << "Finished";
	cout << fds << endl;
	cout << letssee << endl; 
	
	cout << dptr << endl;
	cout << straw << endl;
	cout << straw1 << endl;
	cout << straw2 << endl;
	cout << straw3 << endl;
	cout << straw4 << endl;

	cout << endl; 
	cout << temp; 
	cout << endl; 


	cout << "-------------------------" << endl; 



	uint64_t forfucks = 10; 

	bitset<64> bitset1(forfucks);
	cout << bitset1;










	cout << endl << "FINISH";



	return 0; 
	
	
	


	//return 0; 
}