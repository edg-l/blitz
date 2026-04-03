// EXIT: 0
// OUTPUT: fib(20) = 6765

extern int printf(char* fmt, int a);

int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    printf("fib(20) = %d\n", fib(20));
    return 0;
}
