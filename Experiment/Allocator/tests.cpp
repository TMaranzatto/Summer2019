//#include "TA.h"
//#include "mapalloc.h"
#include "srmap.h"
#include "cll.h"
#include "srmm.h"
#include <iostream>
using namespace std; 





int main() {
	int a = 5;
	int b = 3; 
	int c = 4; 
	int d = 10; 

	srmm Mapping;

	cll l; 
	cout << "The value is" << endl; 
	//cout << tfds << endl;
	l.insert(4);
	srmap Testing;
	void* tfds = Testing.malloc(sizeof(a));
	void* tt = Testing.malloc(sizeof(b));
	l.print(); 
	cout << endl; 
	cout << "------------" << endl;
	size_t sa = sizeof(a);
	size_t sb = sizeof(b);
	Testing.insert(tfds,sa);
	Testing.insert(tt,sb);
	cout << endl; 
	Testing.free(tfds);
	cout << endl; 

	
	//cout << Testing; 

	cout << "--------";
	cout << endl;
	Testing.print();
	cout << endl; 

	//Testing.print();

	//Testing.insert(5);


	size_t j = sizeof(a);
	//mapalloc m(j);

	//int * f = static_cast <int*>(m.malloc(sizeof(b)));
	//int * g = static_cast <int*>(m.malloc(sizeof(c)));
	//int * h = static_cast <int*>(m.malloc(sizeof(d)));

	cout << endl;
	cout << "------------------------------------------------" << endl; 
	
	int lma = 5; 
	int lmb = 4; 

	void* lmaa = Mapping.malloc(sizeof(lma));
	void* lmbb = Mapping.malloc(sizeof(lmb));

	size_t lmaaa = sizeof(lma);
	size_t lmbbb = sizeof(lmb);
	Mapping.insert(lmaa,lmaaa);
	Mapping.insert(lmbb,lmbbb);
	cout << endl; 
	Mapping.free(lmaa);
	
	cout << endl;
	cout << "Its Starting" << endl; 
	Mapping.print();
	cout << endl; 

	
	/*
	Test Mapping 
	
	*/




	cout << endl; 
	cout << "Mapping Complete" << endl; 
	cout << "-------------------------------------------------" << endl; 
	
	//cout << &f << " F" << endl; 
	//cout << &g << " G" << endl; 
	//cout << &h << " D" << endl;

	//allocator a(f); 



	

	//TA test(sizeof(a)); 
	//void* t = test.start; 

	//cout << &t + 512 << "TESTING" << endl; 

	//cout << &t << endl;
	 
	//allocator a; 

	//int* p = static_cast <int*>(test.malloc(a));
	
	
	//cout << "HAHA" << endl; 
	//int s = t; 
	//int ff = p; 

	//cout << t - p << endl;
	//cout << &p;
	cout << "OBJEXT HAS BEEN BUILT";



	//cout << test.start + test.next;
	//cout << p; 
	cout << endl; 
	//cout << test.HeapSize;
	cout << endl; 

	//cout << test.next;


}