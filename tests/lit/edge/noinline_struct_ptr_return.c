// EXIT: 10
// BUG: noinline function returning struct pointer -- the function parameter
// in rdi gets clobbered by the malloc call's argument (also rdi). The
// register allocator should save the parameter across the call but doesn't.

extern char* malloc(int size);

struct Node {
    int value;
    struct Node* next;
};

__attribute__((noinline)) struct Node* new_node(int val) {
    struct Node* n = (struct Node*)malloc(16);
    n->value = val;
    n->next = (struct Node*)0;
    return n;
}

int main() {
    struct Node* head = new_node(10);
    return head->value;
}
