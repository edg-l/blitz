// Test cross-file global variables via extern declarations.
// EXTRA_FILE: helper_counter.c
// OUTPUT: 3

extern int counter;
extern void increment();
extern int printf(char *fmt, int x);

int main() {
    increment();
    increment();
    increment();
    printf("%d\n", counter);
    return 0;
}
