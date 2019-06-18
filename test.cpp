#include "cll.h"
#include "Heap.h"
#include "sort.h"
//#include <iostream>

using namespace std;

int main() {
	//cll <char> test; 
	//new cll<int>; 
	//bool b = test.search("a");
	//test.insert("b");'
	cout << "Hello????";
	cout << "WORKING";
	cout << endl;
	cout << "Test some stuff here";
	//cin >> "hello";
	cout << endl; 
	cll test;

	test.insert(4);
	test.insert(5);

	cout << endl; 
	cout << "searching for 4" << endl; 
	cout << test.search(4);
	cout << endl; 
	test.print();
	cout << endl;

	cout << "Very Nice";
	cout << endl; 
	cout << "PRINTING";
	cout << endl; 
	test.print();
	cout << endl; 
	test.remove();
	cout << endl; 
	test.print();
	cout << endl; 
	cout << "-------------------------------------------------------------------";
	cout << endl << endl << endl; 
	cout << "Begin Heap";
	Heap heapt; 
	heapt.print();
	cout << endl;
	//cout << "------------";
	heapt.insert(4);
	heapt.insert(5);
	cout << endl; 
	heapt.print();
	cout << endl; 
	cout << heapt.contains(4);
	//cout << "-------------------------";
	heapt.remove();
	cout << endl; 
	//cout << "--------------------------------------------------------------";
	cout << endl; 
    heapt.print();
	heapt.insert(3);
	heapt.insert(3);
	heapt.insert(3);
	heapt.insert(3);
	heapt.insert(4);
	//Proper Resize Working
	cout << endl; 
	heapt.print();
	cout << endl; 
	cout << "_------------------------------";
	cout << endl; 
	cout << "Begin Sort";
	sort st; 
	st.insert(4);
	st.print();
	cout << "test" << endl; 
	st.insert(5);
	cout << "test" << endl;
	st.print();
	st.insert(2);
	st.insert(1);
	st.insert(88);
	st.insert(9);
	cout << endl; 
	st.contains(88);
	cout << endl; 

	st.remove(88);
	st.print();
	cout << endl; 






	cout << "--------------------------------------------------------------------------";
	cout << endl; 
	//system("pause");
	return 0;
}