#include <iostream>
#include "MegaAlloc.h"
#include "ultra.h"
#include <conio.h> //for getch()
#include <string>
#include <cmath>
#include <bitset>
#include "superultra.h"
using namespace std; 


int main() {
	MegaAlloc<int> s; 
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
	void* letssee = s.bitallocate(sizeof(b));
	void* strt = s.getstart();
	size_t temp = (size_t)strt - (size_t)fds; 
	cout << endl; 
	cout << endl; 
	cout << "PRINT"<< endl;
	s.print();
	cout << endl; 
	cout << endl; 
	cout << endl; 
	cout << endl; 
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
	cout << endl; 
	cout << endl; 
	superultra<int> su; 
	cout << "SRART SU SHOULD HAVE 1 ARENA" << endl;
	int just = 10; 
	int justs = 10;
	int justss = 10;
	int justsss = 10;
	int jut = 10;
	int juts = 10;
	int jutss = 10;
	int jutsss = 10;
	size_t ff = 10; 
	size_t fff = 10;
	size_t ffff = 64; 
	size_t lfs = 10; 
	su.bitallocate(sizeof(just));
	su.bitallocate(sizeof(justs));
	su.bitallocate(sizeof(justss));
	su.bitallocate(sizeof(justsss));
	su.bitallocate(sizeof(jut));
	su.bitallocate(sizeof(juts));
	su.bitallocate(sizeof(jutss));
	su.bitallocate(sizeof(jutsss));
	su.bitallocate(ff);
	su.bitallocate(fff);
	su.bitallocate(ffff);
	su.bitallocate(lfs);
	void* tes = su.gethead();
	su.deallocate(tes);
	cout << endl; 
	su.print();
	cout << endl << "FINISH";
	return 0;  
}