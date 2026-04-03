// EXIT: 10
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # new_node
// param saved to callee-saved register before call
// CHECK: mov    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: call
// store param value (not clobbered malloc arg)
// CHECK: mov    DWORD PTR

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
