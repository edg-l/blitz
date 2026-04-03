// EXIT: 0
// OUTPUT: counter = 0
// OUTPUT: counter = 5
// OUTPUT: counter = 15
// OUTPUT: counter = 30

extern int printf(char* fmt, int a);

int counter = 0;

__attribute__((noinline)) void add(int n) {
    counter = counter + n;
}

int main() {
    printf("counter = %d\n", counter);
    add(5);
    printf("counter = %d\n", counter);
    add(10);
    printf("counter = %d\n", counter);
    add(15);
    printf("counter = %d\n", counter);
    return 0;
}
