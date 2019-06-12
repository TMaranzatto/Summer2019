
#include <cstddef>
#include <atomic>

struct node {
public:
	node* next;
	int keyvalue;
	bool mark;
	//key = std::hash<std::T>()(data);
	node(node* nxt, int kv) {
		next = nxt;
		keyvalue = kv;
		mark = false;
	}
};

class Window {
public:
	node* pred;
	node* curr;
	Window(node* myPred, node* myCurr) {
		pred = myPred; curr = myCurr;
	}
};

Window* find(node* head, int key) {
	node* pred = NULL;
	node* curr = NULL;
	node* succ = NULL;
	bool marked[1] = { false };
	bool snip;
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
				//pretty sure right now the control flow is blocking
				//code from executing here.  Once we have atomic updates
				//this should be executable
				return new Window(pred, curr);
			}
			pred = curr;
			curr = succ;
		}
	}
}