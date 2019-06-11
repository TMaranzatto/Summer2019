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
 
 Lock-free skip list for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
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

#include"cutil.h"		// Comment this if cutil.h is not available
#include"cuda_runtime.h"
#include"stdio.h"
#include"stdlib.h"
#include"time.h"
#include"assert.h"

#if __WORDSIZE == 64
typedef unsigned long long LL;
#else
typedef unsigned int LL;
#endif

// Maximum level of a node in the skip list
#define MAX_LEVEL 32

// Number of threads per block
#define NUM_THREADS 512

// Supported operations
#define ADD (0)
#define DELETE (1)
#define SEARCH (2)

class Node;

// Definition of generic node class

class __attribute__((aligned (16))) Node
{
  public:
    int topLevel;		// Level of the node
    LL key;			// Key value
    LL next[MAX_LEVEL+1];	// Array of next links

    // Create a next field from a reference and mark bit
    __device__ __host__ LL CreateRef(Node* ref, bool mark)
    {
      LL val=(LL)ref;
      val=val|mark;
      return val;
    }

    __device__ __host__ void SetRef(int index, Node* ref, bool mark)
    {
      next[index]=CreateRef(ref, mark);
    }

    // Extract the reference from a next field
    __device__ Node* GetReference(int index)
    {
      LL ref=next[index];
      return (Node*)((ref>>1)<<1);
    }

    // Extract the reference and mark bit from a next field
    __device__ Node* Get(int index, bool* marked) 
    {
      marked[0]=next[index]%2;
      return (Node*)((next[index]>>1)<<1);
    }

    // CompareAndSet wrapper
    __device__ bool CompareAndSet(int index, Node* expectedRef, Node* newRef, bool oldMark, bool newMark) 
    {
      LL oldVal = (LL)expectedRef|oldMark;
      LL newVal = (LL)newRef|newMark;
      LL oldValOut=atomicCAS(&(next[index]), oldVal, newVal);
      if (oldValOut==oldVal) return true;
      return false;
    }

    // Constructor for sentinel nodes
    Node(LL k)
    {
      key=k;
      topLevel=MAX_LEVEL;
      int i;
      for(i=0;i<MAX_LEVEL+1;i++){
        next[i]=CreateRef((Node*)NULL, false);
      }
    }
};

// Definition of lock-free skip list

class LockFreeSkipList
{
  public:
    Node* head;
    Node* tail;
    LockFreeSkipList()
    {
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
      int i;
      for(i=0;i<h->topLevel+1;i++){
        h->SetRef(i, tail, false);
      }
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
    __device__ bool find(LL, Node**, Node**);	// Helping method
    __device__ bool Add(LL);
    __device__ bool Delete(LL);
    __device__ bool Search(LL);
};

__device__ Node** nodes;			// Pool of pre-allocated nodes
__device__ unsigned int pointerIndex=0;   	// Index into pool of free nodes
__device__ LL* randoms;         		// Array storing the levels of the nodes in the free pool

// Function for creating a new node when requested by an add operation

__device__ Node* GetNewNode(LL key)
{
  LL ind=atomicInc(&pointerIndex, NUM_ITEMS);
  Node* n=nodes[ind];
  n->key=key;
  n->topLevel=randoms[ind];
  int i;
  for(i=0;i<n->topLevel+1;i++){
    n->SetRef(i, NULL, false);
  }
  return n;
}

__device__ LockFreeSkipList* l;		// The lock-free skip list

// Kernel for initializing device memory

__global__ void init(LockFreeSkipList* l1, Node** n, LL* rands)
{
  randoms=rands;
  nodes=n;
  l=l1;
}

// Find the window holding key
// On the way clean up logically deleted nodes (those with set marked bit)

__device__ bool 
LockFreeSkipList::find(LL key, Node** preds, Node** succs)
{ // preds and succs are arrays of pointers
  int bottomLevel=0;
  bool marked[]={false};
  bool snip;
  Node* pred=NULL;
  Node* curr=NULL;
  Node* succ=NULL;
  bool beenThereDoneThat;
  while(true){
    beenThereDoneThat = false;
    pred=head;
    int level;
    for(level=MAX_LEVEL;level>=bottomLevel;level--){
      curr=pred->GetReference(level);
      while(true){
        succ=curr->Get(level, marked);
        while(marked[0]){
          snip=pred->CompareAndSet(level, curr, succ, false, false);
          beenThereDoneThat = true;
          if(!snip) break;
          curr=pred->GetReference(level);
          succ=curr->Get(level, marked);
          beenThereDoneThat = false;
        }
        if (beenThereDoneThat && !snip) break;
        if(curr->key<key){
          pred=curr;
          curr=succ;
        }
        else{
          break;
        }
      }
      if (beenThereDoneThat && !snip) break;
      preds[level]=pred;
      succs[level]=curr;
    }
    if (beenThereDoneThat && !snip) continue;
    return((curr->key==key));
  }
}

__device__ bool
LockFreeSkipList::Search(LL key)
{
  int bottomLevel=0;
  bool marked=false;
  Node* pred=head;
  Node* curr=NULL;
  Node* succ=NULL;
  int level;
  for(level=MAX_LEVEL;level>=bottomLevel;level--){
    curr=pred->GetReference(level);
    while(true){
      succ=curr->Get(level, &marked);
      while(marked){
        curr=curr->GetReference(level);
        succ=curr->Get(level, &marked);
      }
      if(curr->key<key){
        pred=curr;
        curr=succ;
      }
      else{
        break;
      }
    }
  }
  return(curr->key==key);
}

__device__ bool
LockFreeSkipList::Delete(LL key)
{
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  Node* succ;
  bool marked[]={false};
  while(true){
    bool found=find(key, preds, succs);
    if(!found){
      return false;
    }
    else{
      Node* nodeToDelete=succs[bottomLevel];
      int level;
      for(level=nodeToDelete->topLevel;level>=bottomLevel+1;level--){
        succ=nodeToDelete->Get(level, marked);
        while(marked[0]==false){
          nodeToDelete->CompareAndSet(level, succ, succ, false, true);
          succ=nodeToDelete->Get(level, marked);
        }
      }
      succ=nodeToDelete->Get(bottomLevel, marked);
      while(true){
        bool iMarkedIt=nodeToDelete->CompareAndSet(bottomLevel, succ, succ, false, true);
        succ=succs[bottomLevel]->Get(bottomLevel, marked);
        if(iMarkedIt==true){
          find(key, preds, succs);
          return true;
        }
        else if(marked[0]==true){
          return false;
        }
      }
    }
  }
}

__device__ bool
LockFreeSkipList::Add(LL key)
{
  Node* newNode=GetNewNode(key);
  int topLevel=newNode->topLevel;
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  while(true){
    bool found=find(key, preds, succs);
    if(found){
      return false;
    }
    else{
      int level;
      for(level=bottomLevel;level<=topLevel;level++){
         Node* succ=succs[level];
         newNode->SetRef(level, succ, false);
      }
      Node* pred=preds[bottomLevel];
      Node* succ=succs[bottomLevel];
      bool t;
       
      t=pred->CompareAndSet(bottomLevel, succ, newNode, false, false);
      if(!t){
        continue;
      }
      for(level=bottomLevel+1;level<=topLevel;level++){
        while(true){
          pred=preds[level];
          succ=succs[level];
          if(pred->CompareAndSet(level, succ, newNode, false, false)){
            break;
          }
          find(key, preds, succs);
        }
      }
      return true;
    }
  }
}

__global__ void print()
{
  // For debugging
  int tid=blockIdx.x*blockDim.x+threadIdx.x;
  if(tid==0){
    Node* p=l->head;
    bool marked=false;
    while(p!=NULL){
#if __WORDSIZE == 64
      printf("%#llx, %u, marked=%u, address is %p\n", p->key, p->topLevel, marked, p);
#else
      printf("%#x, %u, marked=%u, address is %p\n", p->key, p->topLevel, marked, p);
#endif
      p=p->Get(0, &marked);
    }
    printf("\n");
  }
}

// The main kernel

__global__ void kernel(LL* items, LL* op, LL* result)
{
  // The array items holds the sequence of keys
  // The array op holds the sequence of operations
  // The array result, at the end, will hold the outcome of the operations

  int tid,i;
  for(i=0;i<FACTOR;i++){  // FACTOR is the number of operations per thread
    tid=i*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=NUM_ITEMS) return;

    // Grab the operation and the associated key and execute
    LL item=items[tid];
    if(op[tid]==ADD){
      result[tid]=l->Add(item);
    }
    if(op[tid]==DELETE){
      result[tid]=l->Delete(item);
    }
    if(op[tid]==SEARCH){
      result[tid]=l->Search(item);
    }
  }
}

// Generate the level of a newly created node
LL Randomlevel()
{
  LL v=1;
  double p=0.5;
  while(((rand()/(double)(RAND_MAX))<p) && (v<MAX_LEVEL)) v++;
  return v;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
     printf("Need two arguments: percent add ops and percent delete ops (e.g., 30 50 for 30%% add and 50%% delete).\nAborting...\n");
     exit(1);
  }

  // Extract operations ratio
  int adds=atoi(argv[1]);
  int deletes=atoi(argv[2]);

  if (adds+deletes > 100) {
     printf("Sum of add and delete precentages exceeds 100.\nAborting...\n");
     exit(1);
  }

  // Allocate necessary arrays
  LL* op=(LL*)malloc(sizeof(LL)*NUM_ITEMS);
  LL* levels=(LL*)malloc(sizeof(LL)*NUM_ITEMS);
  LL* items=(LL*)malloc(sizeof(LL)*NUM_ITEMS);
  LL* result=(LL*)malloc(sizeof(LL)*NUM_ITEMS);
  int i;

  // NUM_ITEMS is the total number of operations to execute
  srand(0);
  for(i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;	// Keys associated with operations
  }

  // Pre-generated levels of skip list nodes (relevant only if op[i] is add)
  srand(0);
  for(i=0;i<NUM_ITEMS;i++){
     levels[i]=Randomlevel()-1;
  }

  // Populate the sequence of operations
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
  LL* Clevels;

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMalloc((void**)&Clevels, sizeof(LL)*NUM_ITEMS));
  CUDA_SAFE_CALL(cudaMemcpy(Clevels, levels, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS);
  cudaMalloc((void**)&Clevels, sizeof(LL)*NUM_ITEMS);
  cudaMemcpy(Clevels, levels, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
  cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
  cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif

  Node** pointers=(Node**)malloc(sizeof(LL)*adds);
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
  CUDA_SAFE_CALL(cudaMemcpy(Cpointers,pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds);
  cudaMemcpy(Cpointers,pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice);
#endif
  
  // Allocate the skip list

  LockFreeSkipList* Clist;
  LockFreeSkipList* list=new LockFreeSkipList();
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Clist, sizeof(LockFreeSkipList)));
  CUDA_SAFE_CALL(cudaMemcpy(Clist, list, sizeof(LockFreeSkipList), cudaMemcpyHostToDevice));
#else
  cudaMalloc((void**)&Clist, sizeof(LockFreeSkipList));
  cudaMemcpy(Clist, list, sizeof(LockFreeSkipList), cudaMemcpyHostToDevice);
#endif

  // Calculate the number of thread blocks
  // NUM_ITEMS = total number of operations to execute
  // NUM_THREADS = number of threads per block
  // FACTOR = number of operations per thread

  int blocks=(NUM_ITEMS%(NUM_THREADS*FACTOR)==0)?NUM_ITEMS/(NUM_THREADS*FACTOR):(NUM_ITEMS/(NUM_THREADS*FACTOR))+1;

  // Error checking code
  cudaError_t error=cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error0:CUDA ERROR (%d) {%s}\n",error,cudaGetErrorString(error));
    exit(-1);
  }

  // Initialize the device memory
  init<<<1,32>>>(Clist, Cpointers, Clevels);
  cudaThreadSynchronize();

  // Launch main kernel

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  kernel<<<blocks,NUM_THREADS>>>(Citems, Cop, Cresult);
  
  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print kernel execution time in milliseconds

  printf("%lf\n",time);

  // Check for errors

  error=cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error1:CUDA ERROR (%d) {%s}\n",error, cudaGetErrorString(error));
    exit(-1);
  }

  // Move results back to host memory

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost));
#else
  cudaMemcpy(result, Cresult, sizeof(LL)*NUM_ITEMS, cudaMemcpyDeviceToHost);
#endif

  // Uncomment the following for debugging
  //print<<<1,32>>>();

  cudaThreadSynchronize();
  return 0;
}
