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

 Lock-free priority queue for CUDA; tested for CUDA 4.2 on 32-bit Ubuntu 10.10 and 64-bit Ubuntu 12.04.
 Developed at IIT Kanpur.

 Inputs: Percentage of add operations (e.g., 30 for 30% add)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -arch sm_20 -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -DNUM_ITEMS=num_ops -DFACTOR=num_ops_per_thread -DKEYS=num_keys

 NUM_ITEMS is the total number of operations (mix of add and deleteMin) to execute.

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
#define DELETE_MIN (1)

class Node;

// Definition of generic node class

class __attribute__((aligned (16))) Node
{
  public:
    int topLevel;		// Level of the node
    LL priority;		// Priority value of the node (this is the key value)
    LL marked;			// Special mark field used by DeleteMin
    LL next[MAX_LEVEL+1];	// Array of next links

    // Create a next field from a reference and mark bit
    __device__ __host__ LL CreateRef( Node* ref, bool mark)
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
      priority=k;
      topLevel=MAX_LEVEL;
      marked =0;
      int i;
      for(i=0;i<MAX_LEVEL+1;i++){
        next[i]=CreateRef((Node*)NULL, false);
      }
    }
};


// Definition of a lock-free skip list

class PrioritySkipList
{
  public:
    Node* head;			// Head sentinel
    Node* tail;			// Tail sentinel
    PrioritySkipList()
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

    // Method used by DeleteMin
    __device__ Node* FindAndMarkMin()
    {
      Node* curr =NULL;
      curr=head->GetReference(0);
      while(curr!=tail){
        if(!curr->marked){
          if(0==atomicCAS((LL*)&(curr->marked), 0, 1)) return curr;
        }
        else curr=curr->GetReference(0);
      }
      return NULL;
    }
    __device__ bool find(LL, Node**, Node**);		// Helping method
    __device__ bool Add(Node*);
    __device__ bool Delete(LL);
};

__device__ Node** nodes;			// Pool of pre-allocated nodes
__device__ unsigned int pointerIndex=0; 	// Index into pool of free nodes
__device__ LL* randoms;				// Array storing the levels of the nodes in the free pool

// Function for creating a new node when requested by an add operation

__device__ Node* GetNewNode(LL priority)
{
  LL ind=atomicInc(&pointerIndex, NUM_ITEMS);
  Node* n=nodes[ind];
  n->marked=0;
  n->priority=priority;
  n->topLevel=randoms[ind];
  int i;
  for(i=0;i<n->topLevel+1;i++){
    n->SetRef(i, NULL, false);
  }
  return n;
}

// Definition of a priority queue based on a lock-free skip list

class SkipQueue
{
  public:
    PrioritySkipList* skipList;		// The priority queue
    SkipQueue(){
      skipList=NULL;
    }

    // The method for adding new nodes
    __device__ bool Add(LL item)
    {
      Node* newNode=GetNewNode(item);
      return skipList->Add(newNode);
    }

    // The method for deleting the minimum
    __device__ LL DeleteMin()
    {
      Node* n=skipList->FindAndMarkMin();
      if(n!=NULL){
        skipList->Delete(n->priority);
        return n->priority;
      }
      return 0;
    }
};

__device__ SkipQueue* l;		// The lock-free priority queue

// Kernel for initializing device memory

__global__ void init(SkipQueue* l1, Node** n, LL* levels)
{
  randoms=levels;
  nodes=n;
  l=l1;
}

// Find the window of node with key=priority
// On the way clean up logically deleted nodes (those with set marked bit)

__device__ bool 
PrioritySkipList::find(LL priority, Node** preds, Node** succs)
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
        if(curr->priority<priority){
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
    return((curr->priority==priority));
  }
}

// Called by DeleteMin

__device__ bool
PrioritySkipList::Delete(LL priority)
{
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  Node* succ;
  bool marked[]={false};
  while(true){
    bool found=find(priority, preds, succs);
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
           find(priority, preds, succs);
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
PrioritySkipList::Add(Node* newNode)
{
  LL priority=newNode->priority;
  int topLevel=newNode->topLevel;
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  while(true){
    bool found=find(priority, preds, succs);
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
          find(priority, preds, succs);
        }
      }
      return true;
    }
  }
}

// The main kernel

__global__ void kernel(LL* items, LL* op, LL* result)
{
  // The array items holds the sequence of keys
  // The array op holds the sequence of operations
  // The array result, at the end, will hold the outcome of the operations

  int tid,i;
  for(i=0;i<FACTOR;i++){		// FACTOR is the number of operations per thread
    tid=i*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>NUM_ITEMS) return;

    // Grab the operation and the associated key and execute
    if(op[tid]==ADD){
      result[tid]=l->Add(items[tid]);
    }
    if(op[tid]==DELETE_MIN){
      result[tid]=l->DeleteMin();
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

int main(int argc,char** argv)
{
  if (argc != 2) {
     printf("Need one argument: percent add ops (e.g., 30 for 30%% add ops).\nAborting...\n");
     exit(1);
  }

  if (atoi(argv[1]) > 100) {
     printf("Input more than 100%%.\nAborting...\n");
     exit(1);
  }

  int adds=(NUM_ITEMS*atoi(argv[1]))/100;

  LL op[NUM_ITEMS];		// Sequence of operations
  LL items[NUM_ITEMS];		// Sequence of keys (relevant only if op is add)
  LL result[NUM_ITEMS];		// Sequence of outcomes
  LL levels[NUM_ITEMS];		// Pre-generated levels of newly created nodes (relevant only if op is add)

  // Populate sequence of operations and priorities
  int i;
  srand(0);
  for(i=0;i<adds;i++){
    op[i]=ADD;
    items[i]=10+rand()%KEYS;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=DELETE_MIN;
  }

  // Pre-generate levels of newly created nodes (relevant only if op[i] is add)
  srand(0);
  for(i=0;i<NUM_ITEMS;i++){
    levels[i]=Randomlevel()-1;
  }

  // Allocate device memory

  LL* Citems;
  LL* Cop;
  LL* Cresult;
  LL* Clevels;
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS));
#else
  cudaMalloc((void**)&Cresult, sizeof(LL)*NUM_ITEMS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS));
#else
  cudaMalloc((void**)&Citems, sizeof(LL)*NUM_ITEMS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS));
#else
  cudaMalloc((void**)&Cop, sizeof(LL)*NUM_ITEMS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Clevels, sizeof(LL)*NUM_ITEMS));
#else
  cudaMalloc((void**)&Clevels, sizeof(LL)*NUM_ITEMS);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Clevels, levels, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Clevels, levels, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Citems, items, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Cop, op, sizeof(LL)*NUM_ITEMS, cudaMemcpyHostToDevice);
#endif

  Node* pointers[adds];
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
#else
  cudaMalloc((void**)&Cpointers, sizeof(Node*)*adds);
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Cpointers, pointers, sizeof(Node*)*adds, cudaMemcpyHostToDevice);
#endif

  // Allocate the skip list

  PrioritySkipList* Clist;
  PrioritySkipList* list=new PrioritySkipList();
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Clist, sizeof(PrioritySkipList)));
#else
  cudaMalloc((void**)&Clist, sizeof(PrioritySkipList));
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Clist, list, sizeof(PrioritySkipList), cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Clist, list, sizeof(PrioritySkipList), cudaMemcpyHostToDevice);
#endif

  // Allocate the priority queue

  SkipQueue* Cq;
  SkipQueue* q=new SkipQueue();
  q->skipList=Clist;
#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMalloc((void**)&Cq, sizeof(PrioritySkipList)));
#else
  cudaMalloc((void**)&Cq, sizeof(PrioritySkipList));
#endif

#ifdef _CUTIL_H_
  CUDA_SAFE_CALL(cudaMemcpy(Cq, q, sizeof(PrioritySkipList), cudaMemcpyHostToDevice));
#else
  cudaMemcpy(Cq, q, sizeof(PrioritySkipList), cudaMemcpyHostToDevice);
#endif

  // Calculate the number of thread blocks
  // NUM_ITEMS = total number of operations to execute
  // NUM_THREADS = number of threads per block
  // FACTOR = number of operations per thread

  int blocks=(NUM_ITEMS%(NUM_THREADS*FACTOR)==0)?NUM_ITEMS/(NUM_THREADS*FACTOR):(NUM_ITEMS/(NUM_THREADS*FACTOR))+1;

  // Error checking code
  cudaError_t error= cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error0:CUDA ERROR (%d) {%s}\n",error,cudaGetErrorString(error));
    exit(-1);
  }

  // Initialize the device memory

  init<<<1,32>>>(Cq,Cpointers,Clevels);  
  cudaThreadSynchronize();

  // Launch main kernel

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  kernel<<<blocks,NUM_THREADS>>>(Citems, Cop, Cresult);
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Print kernel execution time in milliseconds

  printf("%lf\n",time);

  // Check for errors

  error= cudaGetLastError();
  if(cudaSuccess!=error){
    printf("error1:CUDA ERROR (%d) {%s}\n",error,cudaGetErrorString(error));
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
