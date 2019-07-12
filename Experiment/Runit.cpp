#include <iostream>
#include "CrashRecoveryBitMap.h"
#include "srmap.h"
#include "bitmaps.h"
#include <vector>

using namespace std; 
vector<size_t> holdsizes;
vector<void*> holdptrs;


int main() {
	
	
	int a = 5; 
	int b = 6;  
	int d = 8; 
	int gggg = 5; 
	srmap Testing;
	void* tfds = Testing.malloc(sizeof(a));
	void* tts = Testing.malloc(sizeof(b));
	void* tst = Testing.malloc(sizeof(d));
	void* tgg = Testing.malloc(sizeof(gggg));
	cout << "------------" << endl;

	size_t sd = sizeof(d);
	size_t sa = sizeof(a);
	size_t sb = sizeof(b);
	size_t sggg = sizeof(gggg);
	cout << tts << endl << "void * ";
	//Insert the first couple of values
	Testing.insert(tfds, sa);
	Testing.insert(tts, sb);
	Testing.insert(tst, sd);
	Testing.insert(tgg, sggg);
	cout << "Insert finished";


	cout << "********************************************************" << endl; 
	Testing.print();
	cout << endl; 

	size_t t = 8;
	void* tt = Testing.malloc(t);
	void* ttt = malloc(t);
	void* tttt = &ttt + 1;
	cout << endl; 

	cout << "HERE IT IS " << endl; 
	cout << tttt; 
	
	//void* tt = malloc(sizeof(t)); 
	//void* ttt = malloc(t);
	CrashRecoveryBitMap m(sa,tfds); 
	cout << endl; 

	//cout << tt; 
	cout << endl; 
	//cout << ttt; 
	cout << "************************************88" << endl; 
	
	Testing.print();
	holdptrs = Testing.voidprint();
	holdsizes = Testing.sizeprint();

	cout << endl; 
	cout << tt;
	cout << endl; 
	cout << ttt; 
	cout << endl;
	cout << "******************************************8";
	
	m.flipbits(2);
	cout << endl; 
	m.print();
	cout << endl; 
	cout << "Best fit is " <<  m.nextfit();
	cout << endl; 
	cout << m.findpos(3);
	cout << endl << endl << endl << endl; 

	cout << "_____________________________" << endl; 
	for (int i = 0; i < holdptrs.size(); i++) {

		cout << holdptrs[i];
		cout << endl; 
		cout << holdsizes[i];
		cout << endl; 


	}

	cout << "_______________________________" << endl; 
	cout << endl; 

	Testing.print();
	int g = (int)holdptrs[1];
	int f = 64 * 32000; 
	int gg = g / f; 
	cout << endl; 
	cout << g; 
	cout << endl; 
	cout << endl; 
	//m.recovery(holdsizes, holdptrs);
	cout << endl; 
	cout << endl; 
	cout << "88888888888888888888";
	cout << endl; 
	//cout << gg; 
	cout << endl; 
	cout << endl; 
	cout << endl; 
	m.print();


	cout << endl; 
	cout << endl; 
	cout << endl; 
	cout << endl; 


	void* vptrs = &ttt + 100; 


	void* Uvptrs = malloc(10);
	void* nuptrs = &Uvptrs + 1; 
	cout << endl; 
	cout << &nuptrs + 10; 
	cout << endl; 

	void* fff = &nuptrs + 10;
	cout << (int)fff; 
	cout << endl; 
	int* iptr = (int*)fff;
	cout << *iptr << " Testing this new iptr";
	
	cout << endl; 
	cout << INT_MIN; 
	int* tiptr = (int*)Uvptrs; 
	cout << endl;
	cout << tiptr << "SHould be some value stored here"; 
	cout << endl; 
	cout << "What i Want to know ";
	cout << fff; 
	cout << "Gotta know";


	cout << endl; 
	cout << endl; 
	cout << Uvptrs; 
	cout << endl; 
	cout << nuptrs; 

	cout << endl; 
	cout << endl; 
	cout << endl; 
	cout << endl; 

	

	cout << endl; 
	cout << f; 


	cout << "_____________________" << endl; 
	void* arena = malloc(80);
	size_t of = 4; 
	size_t max = 80;
	//vector<bool> hehe = m.realrecover(arena, of, max);
	m.print1();



	//cout << "Program Terminated";
	//int dd = 5; 
	//size_t sdd = sizeof(dd);

	//void* testa = Testing.malloc(sdd);
	
	//int a  = 5; 
	//void* test1 = Testing.malloc(sizeof(a));
	
	//cout << endl; 
	//cout << testa; 
	//void *  b = ttt - tt; 

	//int g = static_cast<int>(tt); 

	/*
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
	*/




	return 0;
};