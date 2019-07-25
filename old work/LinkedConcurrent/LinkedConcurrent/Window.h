
#include <cstddef>
#include <atomic>

//some macros to help us
#define N = 1<<20
#define BLOCK_SIZE = 256
#define NBLOCKS = (N + BLOCK_SIZE - 1)/BLOCK_SIZE

struct node {
public:
	node* next;
	int keyvalue;
	bool* mark;
	//key = std::hash<std::T>()(data);
	node(node* nxt, int kv) {
		next = nxt;
		keyvalue = kv;
		mark = false;
	}
};

struct Window {
public:
	node* pred;
	node* curr;
	Window(node* myPred, node* myCurr) {
		pred = myPred; curr = myCurr;
	}
};

__global__
Window* find(node* head, int key) {
	//setting predecessor, current, and successor node
	node* pred;
	node* curr; 
	node* succ;

	cudaMallocManaged(&pred, sizeof(pred));
	cudaMallocManaged(&curr, sizeof(curr));
	cudaMallocManaged(&succ, sizeof(succ));

	pred = NULL; curr = NULL; succ = NULL;

	//setting the marked boolean singleton and snip boolean

	bool* marked[1];
	bool* snip;
	cudaMallocManaged(marked, sizeof(marked));

	marked[0] = false;

	label: while (true) {
		pred = head;
		curr = pred->next;
		while (true) {

			//fill with get(marked) once implemented
			//should be atomic
			succ = curr->next;
			marked[0] = succ->mark;



			while (marked[0]) {

				//fill with compare and set
				snip = pred->mark;


				if (!snip) goto label;
				curr = succ;

				//replace with get statement
				//agian, should be atomic
				succ = curr -> next;
				marked[0] = succ->mark;


			}
			if (curr -> keyvalue >= key) {
				cudaFree(snip);
				cudaFree(marked);
				cudaFree(succ);

				Window* ret = new Window(pred, curr);

				cudaFree(pred);
				cudaFree(curr);
				return ret;
			}
			pred = curr;
			curr = succ;
		}
	}
}