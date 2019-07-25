/*

Copyright 2012-2013 Indian Institute of Technology Kanpur. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions, and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY INDIAN INSTITUTE OF TECHNOLOGY KANPUR ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL INDIAN INSTITUTE OF TECHNOLOGY KANPUR OR
THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied, of Indian Institute of Technology Kanpur.

*/

/**********************************************************************************

 Lock-free hash table for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
 Developed at IIT Kanpur.

 Inputs: Percentage of add and delete operations (e.g., 30 50 for 30% add and 50% delete)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -arch sm_20 -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -DNUM_ITEMS=num_ops -DFACTOR=num_ops_per_thread -DKEYS=num_keys

 NUM_ITEMS is the total number of operations (mix of add, delete, search) to execute.

 FACTOR is the number of operations per thread.

 KEYS is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen as zero. Any positive shift other than +10 would also work.

 The include path ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ is needed for cutil.h.

 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

***************************************************************************************/

#include"cutil.h"			// Comment this if cutil.h is not available
#include"cuda_runtime.h"
#include"stdio.h"

#if __WORDSIZE == 64
typedef unsigned long long LL;
#else
typedef unsigned int LL;
#endif

// Number of threads per block
#define NUM_THREADS 512

// Number of hash table buckets
#define NUM_BUCKETS 10000

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

// Definition of generic node class

class __attribute__((aligned (16))) Node
{
  public:
    LL key;
    LL next;

    // Create a next field from a reference and mark bit
    __device__ __host__ LL CreateRef(Node* ref, bool mark)
    {
      LL val=(LL)ref;
      val=val|mark;
      return val;
    }

    __device__ __host__ void SetRef(Node* ref, bool mark)
    {
      next=CreateRef(ref, mark);
    }

    // Extract the reference from a next field
    __device__ Node* GetReference()
    {
      LL ref=next;
      return (Node*)((ref>>1)<<1);
    }

    // Extract the reference and mark bit from a next field
    __device__ Node* Get(bool* marked)
    {
      *marked=next%2;
      return (Node*)((next>>1)<<1);
    }

    // CompareAndSet wrapper
    __device__ bool CompareAndSet(Node* expectedRef, Node* newRef, bool oldMark, bool newMark)
    {
      LL oldVal = (LL)expectedRef|oldMark;
      LL newVal = (LL)newRef|newMark;
      LL oldValOut=atomicCAS(&(next), oldVal, newVal);
      if (oldValOut==oldVal) return true;
      return false;
    }

    // Constructor for sentinel nodes
    Node(LL k)
    {
      key=k;
      next=CreateRef((Node*)NULL, false);
    }
};

__device__ Node** nodes;			// Pool of pre-allocated nodes
__device__ unsigned int pointerIndex=0; 	// Index into pool of free nodes

// Function for creating a new node when requested by an add operation

__device__ Node* GetNewNode(LL key)
{
  LL ind=atomicInc(&pointerIndex, NUM_ITEMS);
  Node* n=nodes[ind];
  n->key=key;
  n->SetRef(NULL, false);
  return n;
}

// Window of node containing a particular key

class Window
{
  public:
    Node* pred;		// Predecessor of node holding the key being searched
    Node* curr;		// The node holding the key being searched (if present)

    __device__ Window(Node* myPred, Node* myCurr)
    {
      pred=myPred;
      curr=myCurr;
    }
};

// Lock-free linked list

class LinkedList
{
  public:
    __device__ void Find(Window*, LL);		// Helping method
    __device__ bool Add(LL);
    __device__ bool Delete(LL);
    __device__ bool Search(LL);

    Node* head;			// Head sentinel
    Node* tail;			// Tail sentinel

    LinkedList(int index)
    {
      if(index==0){
        Node* h=new Node(0);
#if __WORDSIZE == 64
        Node* t=new Node((LL)0xffffffffffffffff);
#else
        Node* t=new Node((LL)0xffffffff);
#endif
#ifdef _CUTIL_H_
        CUDA_SAFE_CALL(cudaMalloc((void**)&head, sizeof(Node)));
#else
        cudaMalloc((void**)&head, sizeof(Node));
#endif

#ifdef _CUTIL_H_
        CUDA_SAFE_CALL(cudaMalloc((void**)&tail, sizeof(Node)));
#else
        cudaMalloc((void**)&tail, sizeof(Node));
#endif
        h->next=(LL)tail;
#ifdef _CUTIL_H_
        CUDA_SAFE_CALL(cudaMemcpy(head, h, sizeof(Node), cudaMemcpyHostToDevice));
#else
        cudaMemcpy(head, h, sizeof(Node), cudaMemcpyHostToDevice);
#endif

#ifdef _CUTIL_H_
        CUDA_SAFE_CALL(cudaMemcpy(tail, t, sizeof(Node), cudaMemcpyHostToDevice));
#else
        cudaMemcpy(tail, t, sizeof(Node), cudaMemcpyHostToDevice);
#endif
      }
      else{
#ifdef _CUTIL_H_
        CUDA_SAFE_CALL(cudaMalloc((void**)&head, sizeof(Node)));
#else
        cudaMalloc((void**)&head, sizeof(Node));
#endif
      }
    }
};

// Find the window holding key
// On the way clean up logically deleted nodes (those with set marked bit)

__device__ void
LinkedList::Find(Window* w, LL key)
{
  Node* pred;
  Node* curr;
  Node* succ;
  bool marked[]={false};
  bool snip;

  retry:
  while(true){
     pred=head;
     curr=pred->GetReference();
     while(true){
        succ=curr->Get(marked);
        while(marked[0]){
           snip=pred->CompareAndSet(curr, succ, false, false);
           if(!snip) goto retry;
           curr=succ;
           succ=curr->Get(marked);
        }
	if (curr->key >= key){
           w->pred=pred;
           w->curr=curr;
	   return;
	}
	pred=curr;
	curr=succ;
     }
  }
}

__device__ bool 
LinkedList::Search(LL key)
{
  bool marked;
  Node* curr = head->GetReference();
  while(curr->key<key){
     curr=curr->Get(&marked);
  }
  return((curr->key==key) && !marked);
}
   
__device__ bool
LinkedList::Delete(LL key)
{
  Window w(NULL, NULL);
  bool snip;
  while(true){
     Find(&w, key);
     Node* curr=w.curr;
     Node* pred=w.pred;
     if(curr->key!=key){
        return false;
     }
     else{
        Node* succ = curr->GetReference();
        snip=curr->CompareAndSet(succ, succ, false, true);
	if(!snip) continue;
	pred->CompareAndSet(curr, succ, false, false);
	return true;
     }
  }
}

__device__ bool
LinkedList::Add(LL key)
{
  Node* pointer=GetNewNode(key);
  Window w(NULL,NULL);
  while(true){
     Find(&w, key);
     Node* pred=w.pred;
     Node* curr=w.curr;
     if(curr->key==key) return false;
     else{
        pointer->key=key;
        pointer->SetRef(curr, false);
        if(pred->CompareAndSet(curr, pointer, false, false)) {
           return true;
        }
     }
  }
}

// Modulo hash function

__device__ LL Hash(LL x)
{
  return x%NUM_BUCKETS;
}

__device__ LinkedList** bucketList;     // List of hash table buckets
                                        // Each bucket is a lock-free linked list

// Kernel for initializing device memory
// This kernel initializes the bucket heads and links them up

__global__ void init(LinkedList** Lists)
{
  bucketList=Lists;
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  tid++;
  if (tid>=NUM_BUCKETS)
    return;
  Node* head=Lists[tid]->head;

#if __WORDSIZE == 64
  LL key=(LL)(0x8000000000000000|tid);
#else
  LL key=(LL)(0x80000000|tid);
#endif

  head->key=key;
  Window w(NULL, NULL);
  while(true){
    Lists[0]->Find(&w, key);
    Node* pred=w.pred;
    Node* curr=w.curr;
    if(curr->key==key) return;
    head->SetRef(curr, false);
    if(pred->CompareAndSet(curr, head, false, false)){
       return ;
    }
  }
}

// The main kernel

__global__ void kernel(LL* items, LL* op, LL* result, Node** n)
{
  // The array items holds the sequence of keys
  // The array op holds the sequence of operations
  // The array result, at the end, will hold the outcome of the operations
  // n points to an array of pre-allocated free linked list nodes

  nodes=n;
  int tid,i;
  for(i=0;i<FACTOR;i++){		// FACTOR is the number of operations per thread
    tid=i*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=NUM_ITEMS) return;

    // Grab the operation and the associated key and execute
    LL itm=items[tid];
    LL bkt=Hash(itm);
    if(op[tid]==ADD){
      result[tid]=bucketList[bkt]->Add(itm);
    }
    if(op[tid]==DELETE){
      result[tid]=bucketList[bkt]->Delete(itm);
    }
    if(op[tid]==SEARCH){
      result[tid]=bucketList[bkt]->Search(itm);
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 3) {
     printf("Need two arguments: percent add ops and percent delete ops (e.g., 30 50 for 30%% add and 50%% delete).\nAborting...\n");
     exit(1);
  }

  int adds=atoi(argv[1]);
  int deletes=atoi(argv[2]);

  if (adds+deletes > 100) {
     printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     exit(1);
  }

  // Allocate hash table

  LinkedList* buckets[NUM_BUCKETS];
  LinkedList** Cbuckets;
  int i;
  for(i=0;i<NUM_BUCKETS;i++){
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMalloc((void**)&(buckets[i]), sizeof(LinkedList)));
#else
    cudaMalloc((void**)&(buckets[i]), sizeof(LinkedList));
#endif
    LinkedList* l=new LinkedList(i);
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMemcpy(buckets[i], l, sizeof(LinkedList), cudaMemcpyHostToDevice));
#else
    cudaMemcpy(buckets[i], l, sizeof(LinkedList), cudaMemcpyHostToDevice);
#endif
  }
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&(Cbuckets), sizeof(LinkedList*)*NUM_BUCKETS));
#else
  cudaMalloc((void**)&(Cbuckets), sizeof(LinkedList*)*NUM_BUCKETS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Cbuckets, buckets, sizeof(LinkedList*)*NUM_BUCKETS, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Cbuckets, buckets, sizeof(LinkedList*)*NUM_BUCKETS, cudaMemcpyHostToDevice);
#endif

  // Initialize the device memory
  
  int b=(NUM_BUCKETS/32)+1;
  init<<<b, 32>>>(Cbuckets);

  
  LL* op=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Array of operations
  LL* items=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Array of keys
  LL* result=(LL*)malloc(sizeof(LL)*NUM_ITEMS);		// Arrays of outcome

  srand(0);

  // NUM_ITEMS is the total number of operations to execute
  for(i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;	// Keys
  }

  // Populate the op sequence
  for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    op[i]=ADD;
  }
  for(;i<(NUM_ITEMS*(adds+deletes))/100;i++){
    op[i]=DELETE;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=SEARCH;
  }

  adds=(NUM_ITEMS*adds)/100;

  // Allocate device memory

  LL* Citems;
  LL* Cop;
  LL* Cresult;
  
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS);
  cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
  cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif
  Node** pointers=(Node**)malloc(sizeof(Node*)*adds);
  Node** Cpointers;

  // Allocate the pool of free nodes

  for(i=0;i<adds;i++){
#ifdef _CUTIL_H_
    CUDA_SAFE_CALL(cudaMalloc((void**)&pointers[i], sizeof(Node)));
#else
    cudaMalloc((void**)&pointers[i], sizeof(Node));
#endif
  }
  
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds));
  CUDA_SAFE_CALL(cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds);
  cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice);
#endif

  // Calculate the number of thread blocks
  // NUM_ITEMS = total number of operations to execute
  // NUM_THREADS = number of threads per block
  // FACTOR = number of operations per thread

  int blocks=(NUM_ITEMS%(NUM_THREADS*FACTOR)==0)?NUM_ITEMS/(NUM_THREADS*FACTOR):(NUM_ITEMS/(NUM_THREADS*FACTOR))+1;

  // Launch main kernel

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  
  kernel<<<blocks, NUM_THREADS>>>(Citems, Cop, Cresult, Cpointers);
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print kernel execution time in milliseconds

  printf("%lf\n",time);

  // Check for errors

  cudaError_t error = cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error:CUDA ERROR (%d) {%s}\n",error, cudaGetErrorString(error));
    exit(-1);
  }

  // Move results back to host memory

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost));
#else
  cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost);
#endif

  return 0;
}
