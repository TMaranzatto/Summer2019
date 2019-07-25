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

 Lock-free priority queue for POSIX threads
 Developed at IIT Kanpur.

 Inputs: Percentage of add operations (e.g., 30 for 30% add)
 Output: Prints the total time (in milliseconds) to execute the the sequence of operations

 Compilation flags: -O3 -pthread -DNUM_ITEMS=num_ops -DNUM_THREADS=num_threads -DKEYS=num_keys -DPRE_ALLOCATE

 Optional compilation flag: -DPRE_ALLOCATE

 NUM_ITEMS is the total number of operations (mix of add and delete_min) to execute.

 NUM_THREADS is the number of threads.

 KEYS is the number of integer keys assumed in the range [10, 9+KEYS].
 The paper cited below states that the key range is [0, KEYS-1]. However, we have shifted the range by +10 so that
 the head sentinel key (the minimum key) can be chosen from [0, 9]. Any positive shift other than +10 would also work.

 If PRE_ALLOCATE is turned on, all dynamic memory will be allocated before the sequence of operations begins.

 Related work:

 Prabhakar Misra and Mainak Chaudhuri. Performance Evaluation of Concurrent Lock-free Data Structures
 on GPUs. In Proceedings of the 18th IEEE International Conference on Parallel and Distributed Systems,
 December 2012.

***************************************************************************************/

#include"stdio.h"
#include"stdlib.h"
#include"time.h"
#include"pthread.h"
#include"assert.h"
#include"sys/time.h"

#if __WORDSIZE == 64
typedef unsigned long long LL;
#else
typedef unsigned int LL;
#endif

// Maximum level of a node in the skip list
#define MAX_LEVEL 32

// Supported operations
#define ADD (0)
#define DELETE_MIN (1)

LL items[NUM_ITEMS];		// Array of keys associated with operations (relevant only if op is add)
LL op[NUM_ITEMS];		// Array of operations
LL result[NUM_ITEMS];		// Array of outcomes

class __attribute__((aligned (16))) Node;	// Generic node class

class AtomicReference
{
  public:
    LL reference;

    // Create a next field from a reference and mark bit
    AtomicReference(Node* ref, bool mark)
    {
      reference=(LL)(ref)|mark;
    }

    AtomicReference()
    {
      reference=0;
    }

    bool CompareAndSet(Node* expectedRef, Node* newRef, bool oldMark, bool newMark);
    Node* Get(bool* marked);
    void Set(Node* newRef,bool newMark);
    Node* GetReference();
};

// CompareAndSet wrapper
bool
AtomicReference::CompareAndSet(Node* expectedRef, Node* newRef, bool oldMark, bool newMark)
{
  LL oldVal = (LL)expectedRef|oldMark;
  LL newVal = (LL)newRef|newMark;
  LL oldValOut;
  bool result;

#if __WORDSIZE == 64
  asm(
                "lock cmpxchgq %4, %1 \n setzb %0"
                :"=qm"(result),  "+m" (reference), "=a" (oldValOut)
                :"a" (oldVal),  "r" (newVal)
                :
        );
#else
  asm(
		"lock cmpxchgl %4, %1 \n setzb %0"
		:"=qm"(result),  "+m" (reference), "=a" (oldValOut)
		:"a" (oldVal),  "r" (newVal)
		: 
	);
#endif
  return result;
}

// Extract the reference and mark bit from a next field
Node*
AtomicReference::Get(bool* marked)
{
  *marked=reference%2;
  return (Node*)((reference>>1)<<1);
}

void 
AtomicReference::Set(Node* newRef, bool newMark)
{
  reference=(LL)newRef|newMark;
}

// Extract the reference from a next field
Node*
AtomicReference::GetReference()
{
  return (Node*)((reference>>1)<<1);
}

// Definition of generic node class

class __attribute__((aligned (16))) Node
{
  public:
    LL marked;		// Special mark field used by DeleteMin
    int topLevel;	// Level of the node
    LL key;		// Priority value of the node (this is the key value)
    AtomicReference next[MAX_LEVEL+1]; // Array of next links

    Node(LL k, LL height)
    {
      marked=0;
      key=k;
      topLevel=height;

      int i;
      for(i=0;i<topLevel+1;i++){
        next[i].Set(NULL, false);
      }
    }

    Node(LL k)
    {
      marked=0;
      key=k;
      topLevel=MAX_LEVEL;

      int i;
      for(i=0;i<topLevel+1;i++){
        next[i].Set(NULL, false);
      }
    }
};

// An alternate wrapper for CAS

bool
CAS(LL* add, LL old, LL nu)
{
  LL oldValOut;
  bool result;

#if __WORDSIZE == 64
    asm(
                        "lock cmpxchgq %4, %1 \n setzb %0"
                        :"=qm"(result),  "+m" (*add), "=a" (oldValOut)
                        :"a" (old),  "r" (nu)
                        :
                  );
#else
    asm(
			"lock cmpxchgl %4, %1 \n setzb %0"
			:"=qm"(result),  "+m" (*add), "=a" (oldValOut)
			:"a" (old),  "r" (nu)
			: 
		  );
#endif
  return result;
}

// Definition of a lock-free skip list

class PrioritySkipList
{
  private:
    bool find(LL, Node**, Node**);	// Helping method
  public:
    Node* head;			// Head sentinel
    Node* tail;			// Tail sentinel

    PrioritySkipList()
    {
      head=new Node(0);
#if __WORDSIZE == 64
      tail=new Node((LL)0xffffffffffffffff);
#else
      tail=new Node((LL)0xffffffff);
#endif

      int i;
      for(i=0;i<head->topLevel+1;i++){
        head->next[i].Set(tail, false);
      }
    }

    bool Add(LL, Node*);
    bool Delete(LL);

    // Method used by DeleteMin
    Node* FindAndMarkMin()
    {
      Node* curr =NULL;
      curr=head->next[0].GetReference();
      while(curr!=tail){
        if(!curr->marked){
          if(CAS((LL*)&(curr->marked), 0, 1)) return curr;
        }
        else curr=curr->next[0].GetReference();
      }
      return NULL;
    }
};

// Find the window of node with key=priority
// On the way clean up logically deleted nodes (those with set marked bit)

bool 
PrioritySkipList::find(LL key, Node** preds, Node** succs)
{ // preds and succs are arrays of pointers
  int bottomLevel=0;
  bool marked=false;
  bool snip;
  Node* pred=NULL;
  Node* curr=NULL;
  Node* succ=NULL;
  
  retry:
  while (true) {
    pred=head;
    int level;
    for(level=MAX_LEVEL;level>=bottomLevel;level--){
      curr=pred->next[level].GetReference();
      while(true) {
        succ=curr->next[level].Get(&marked);
        while(marked) {
          snip=pred->next[level].CompareAndSet(curr, succ, false, false);
          if (!snip) goto retry;
          curr=pred->next[level].GetReference();
          succ=curr->next[level].Get(&marked);
        }
        if(curr->key<key) {
          pred=curr;
          curr=succ;
        }
        else {
          break;
        }
      }
      preds[level]=pred;
      succs[level]=curr;
    }
    return(curr->key==key);
  }
}

// Generate the level of a newly created node

int randomLevel()
{
  int v=1;
  double p=0.5;
  while(((rand()/(double)(RAND_MAX))<p) && (v<MAX_LEVEL)) v++;
  return v;
}

bool
PrioritySkipList::Add(LL key, Node *n)
{
#ifdef PRE_ALLOCATE
  int topLevel=n->topLevel;
#else
  int topLevel=randomLevel()-1;
#endif
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  while(true) {
    bool found=find(key, preds, succs);
    if(found) {
      return false;
    }
    else{
#ifdef PRE_ALLOCATE
      n->key = key;
      Node* newNode = n;
#else
      Node* newNode=new Node(key, topLevel);
#endif
      int level;
      for(level=bottomLevel;level<=topLevel;level++){
        Node* succ=succs[level];
        newNode->next[level].Set(succ, false);
      }
      Node* pred=preds[bottomLevel];
      Node* succ=succs[bottomLevel];
      newNode->next[bottomLevel].Set(succ, false);
      if(!pred->next[bottomLevel].CompareAndSet(succ, newNode, false, false)) {
        continue;
      }
      for(level=bottomLevel+1;level<=topLevel;level++){
        while(true) {
          pred=preds[level];
          succ=succs[level];
          if(pred->next[level].CompareAndSet(succ, newNode, false, false))
             break;
          find(key, preds, succs);
        }
      }
      return true;
    }
  }
}

// Called by DeleteMin

bool
PrioritySkipList::Delete(LL key)
{
  int bottomLevel=0;
  Node* preds[MAX_LEVEL+1];
  Node* succs[MAX_LEVEL+1];
  Node* succ;
  while (true) {
    bool found=find(key, preds, succs);
    assert(found==true);
    {
      Node* nodeToDelete=succs[bottomLevel];
      assert(nodeToDelete->marked=true);
      int level;
      for(level=nodeToDelete->topLevel;level>=bottomLevel+1;level--){
        bool marked=false;
        succ=nodeToDelete->next[level].Get(&marked);
        while(!marked){
          nodeToDelete->next[level].CompareAndSet(succ, succ, false, true);
          succ=nodeToDelete->next[level].Get(&marked);
        }
      }
      bool marked=false;
      succ=nodeToDelete->next[bottomLevel].Get(&marked);
      while(true){
        bool iMarkedIt=nodeToDelete->next[bottomLevel].CompareAndSet(succ, succ, false, true);
        succ=succs[bottomLevel]->next[bottomLevel].Get(&marked);
        if(iMarkedIt){
          find(key, preds, succs);
          return true;
        }
        else if(marked){
          return false;
        }
      }
    }
  }
}

// Definition of a priority queue based on a lock-free skip list

class SkipQueue
{
  public:
    PrioritySkipList skipList;		// The priority queue

    bool Add(LL item, Node *n)
    {
      return skipList.Add(item, n);
    }

    LL DeleteMin()
    {
      Node* n=skipList.FindAndMarkMin();
      if (n!=NULL) {
        skipList.Delete(n->key);
        return n->key;
      }
      return 0;
    }
};

SkipQueue l;		// The lock-free priority queue

// For debugging

void PrintList()
{
  Node* p=l.skipList.head;
  while(p!=NULL){
#if __WORDSIZE == 64
    printf("%#llx\n", p->key);
#else
    printf("%#x\n", p->key);
#endif
    p=p->next[0].GetReference();
  }
  printf("\n");
}

#ifdef PRE_ALLOCATE
Node ***freelist;                       // Per-thread free pool
unsigned indexPointer[NUM_THREADS];     // Index into free pool
#endif

// The worker thread function
// The thread id is passed as an argument

void* Thread(void* t)
{
  unsigned int tid=(unsigned long) t;
  int i;
  for(i=tid;i<NUM_ITEMS;i+=NUM_THREADS){
    // Grab the operations and execute   
    unsigned int item =items[i];
    switch(op[i]){
      case(ADD):
#ifdef PRE_ALLOCATE
        result[i]=10+l.Add(item, freelist[tid][indexPointer[tid]]);
        indexPointer[tid]++;
#else
        result[i]=10+l.Add(item, NULL);
#endif
        break;
      case(DELETE_MIN):
        result[i]=l.DeleteMin();
        break;
    }
  }
}

int main(int argc,char** argv)
{
  if (argc != 2) {
     printf("Need one argument: percent add ops (e.g., 30 for 30%% add ops).\nAborting...\n");
     exit(1);
  }

  // Extract operations ratio
  int adds=atoi(argv[1]);

  if (adds > 100) {
     printf("Input exceeds 100%%.\nAborting...\n");
     exit(1);
  }

  // Initialize thread stack

  void* status;
  pthread_t threads[NUM_THREADS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr,102400);

  int rc;
  long t;
  int i, j;

#ifdef PRE_ALLOCATE
  // Allocate free pool

  srand(0);
  freelist = new Node**[NUM_THREADS];
  assert(freelist != NULL);
  for (i=0; i<NUM_THREADS; i++) {
     indexPointer[i] = 0;
     freelist[i] = new Node*[(NUM_ITEMS*adds)/(100*NUM_THREADS)+1];
     assert(freelist[i] != NULL);
     for (j=0; j<(NUM_ITEMS*adds)/(100*NUM_THREADS)+1; j++) {
        freelist[i][j] = new Node(0, randomLevel()-1);
        assert(freelist[i][j] != NULL);
     }
  }
#endif

  // Populate key array (relevant if op[i] is add)
  // NUM_ITEMS is the total number of operations
  srand(0);
  for(i=0;i<NUM_ITEMS;i++){
    items[i]=10+rand()%KEYS;		// KEYS is the number of integer keys
  }

  // Populate op array
  for(i=0;i<(NUM_ITEMS*adds)/100;i++){
    op[i]=ADD;
  }
  for(;i<NUM_ITEMS;i++){
    op[i]=DELETE_MIN;
  }
  
  struct timeval tv0,tv1;
  struct timezone tz0,tz1;
  
  gettimeofday(&tv0,&tz0);

  // Spawn threads

  for(t=0;t<NUM_THREADS;t++){
    rc = pthread_create(&threads[t], &attr, Thread, (void *)(t));
    if (rc){
      printf("ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  // Join threads
  
  for(t=0; t<NUM_THREADS; t++) {
    rc = pthread_join(threads[t], &status);
  }
  gettimeofday(&tv1,&tz1);

  // Print time in ms

  printf("%lf\n",((float)((tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec)))/1000.0);
  return 0;
}
