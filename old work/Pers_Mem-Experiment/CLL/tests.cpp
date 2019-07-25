#include "cll.h"
#include <iostream>

int main(){
    cll<int> linkedList;
    bool b = linkedList.search(7);
    linkedList.insert(5);
    return 0;
}