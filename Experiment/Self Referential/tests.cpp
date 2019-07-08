//#include "TA.h"
//#include "mapalloc.h"
#include "srmap.h"
#include "cll.h"
#include "srmm.h"
#include <iostream>
using namespace std; 





int main() {

	//ALlocate some simple variables to store in the Linked List
	int a = 5;
	int b = 3; 
	int c = 4; 
	int d = 10; 

	//Instantiate the MMAP allocator
	srmm Mapping;


	/*
	Basic Linked List Testing
	*/
	cll l; 
	cout << "The value is" << endl; 
	l.insert(4);
	l.print();
	cout << endl;

	//Start testing the Simple Allocator

	srmap Testing;
	void* tfds = Testing.malloc(sizeof(a));
	void* tt = Testing.malloc(sizeof(b));  
	void* tst = Testing.malloc(sizeof(d));
	cout << "------------" << endl;

	size_t sd = sizeof(d);
	size_t sa = sizeof(a);
	size_t sb = sizeof(b);
	cout << tt << endl << "void * ";
	//Insert the first couple of values
	Testing.insert(tfds,sa);
	Testing.insert(tt,sb);
	Testing.insert(tst, sd);
	
	cout << "insert works" << endl;

	//First iteration of the free function... Optimized in MMAP 
	int dd = Testing.free(tt);
	
	
	cout << endl; 
	cout << dd; 
	cout << endl; 
	cout << "Free works";
	
	//Make Sure we are NOT allocating a new block
	void* thisis = Testing.malloc(sizeof(c));


	cout << thisis << endl << "HERE IT IS";



	size_t sc = sizeof(c);
	

	//No reason to insert this into the list since the Node does not leave but testing Linked List
	Testing.insert(thisis, sc);
	cout << "********************************************************************************************************";

	cout << endl; 
	Testing.free(tfds);
	cout << endl; 

	
	//cout << Testing; 

	cout << "--------";
	cout << endl;
	cout << "*****" << endl;
	Testing.print();
	cout << endl; 

	//END SIMPLE ALLOCATOR


	size_t j = sizeof(a);
	//mapalloc m(j);

	//int * f = static_cast <int*>(m.malloc(sizeof(b)));
	//int * g = static_cast <int*>(m.malloc(sizeof(c)));
	//int * h = static_cast <int*>(m.malloc(sizeof(d)));


	//Begin MMAP Allocator
	cout << endl;
	cout << "------------------------------------------------" << endl; 
	
	int lma = 5; 
	int lmb = 4; 
	int lmc = 6; 


	//Test the Malloc
	void* lmaa = Mapping.malloc(sizeof(lma));
	void* lmbb = Mapping.malloc(sizeof(lmb));

	size_t lmaaa = sizeof(lma);
	size_t lmbbb = sizeof(lmb);
	size_t lmccc = sizeof(lmc);
	//Test Inserting into the Linked List Data Structure
	Mapping.insert(lmaa,lmaaa);
	Mapping.insert(lmbb,lmbbb);
	cout << endl; 
	//Test Freeing one of the blocks of memory 
	Mapping.free(lmaa);
	void* lmcc = Mapping.malloc(sizeof(lmc));
	Mapping.insert(lmcc, lmccc);


	//Print the Map of the linked List

	cout << endl;
	cout << "Its Starting" << endl; 
	cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
	Mapping.print();
	cout << endl; 
	cout << endl; 
	cout << "Mapping Complete" << endl; 
	cout << "-------------------------------------------------" << endl; 
	cout << "PROGRAM TERMINATED";



}