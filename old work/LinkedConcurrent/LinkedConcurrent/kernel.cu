
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "NonBlockingLinkedList.h"
#include "kernel.h"

__global__
int main()
{
	NonBlockingLinkedList* l = new NonBlockingLinkedList();

    return 0;
}