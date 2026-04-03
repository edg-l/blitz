// EXIT: 0
// OUTPUT: 10 -> 20 -> 30 -> 40 -> 50 -> end
// OUTPUT: sum = 150
// OUTPUT: count = 5

extern int printf(char* fmt, int a);
extern int puts(char* s);
extern char* malloc(int size);

struct Node {
    int value;
    struct Node* next;
};

// Allocate node inline in main to avoid the noinline param clobber bug
__attribute__((noinline)) void print_list(struct Node* head) {
    struct Node* cur = head;
    while (cur != (struct Node*)0) {
        printf("%d -> ", cur->value);
        cur = cur->next;
    }
    puts("end");
}

__attribute__((noinline)) int sum_list(struct Node* head) {
    int sum = 0;
    struct Node* cur = head;
    while (cur != (struct Node*)0) {
        sum = sum + cur->value;
        cur = cur->next;
    }
    return sum;
}

__attribute__((noinline)) int count_list(struct Node* head) {
    int count = 0;
    struct Node* cur = head;
    while (cur != (struct Node*)0) {
        count = count + 1;
        cur = cur->next;
    }
    return count;
}

int main() {
    // Build list manually to avoid noinline param clobber bug
    struct Node* n5 = (struct Node*)malloc(16);
    n5->value = 50;
    n5->next = (struct Node*)0;

    struct Node* n4 = (struct Node*)malloc(16);
    n4->value = 40;
    n4->next = n5;

    struct Node* n3 = (struct Node*)malloc(16);
    n3->value = 30;
    n3->next = n4;

    struct Node* n2 = (struct Node*)malloc(16);
    n2->value = 20;
    n2->next = n3;

    struct Node* n1 = (struct Node*)malloc(16);
    n1->value = 10;
    n1->next = n2;

    print_list(n1);
    printf("sum = %d\n", sum_list(n1));
    printf("count = %d\n", count_list(n1));
    return 0;
}
