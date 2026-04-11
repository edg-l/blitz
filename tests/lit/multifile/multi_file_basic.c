// Test multi-file compilation: call functions defined in another file.
// EXTRA_FILE: helper_add.c
// OUTPUT: 7
// OUTPUT: 12

extern int add(int a, int b);
extern int multiply(int a, int b);
extern int printf(char *fmt, int x);

int main() {
    printf("%d\n", add(3, 4));
    printf("%d\n", multiply(3, 4));
    return 0;
}
