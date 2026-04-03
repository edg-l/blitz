// EXIT: 0
// OUTPUT: fib(0) = 0
// OUTPUT: fib(1) = 1
// OUTPUT: fib(2) = 1
// OUTPUT: fib(3) = 2
// OUTPUT: fib(4) = 3
// OUTPUT: fib(5) = 5
// OUTPUT: fib(6) = 8
// OUTPUT: fib(7) = 13
// OUTPUT: fib(8) = 21
// OUTPUT: fib(9) = 34
// OUTPUT: fib(10) = 55
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # fib
// recursive calls
// CHECK: call
// CHECK: call
// CHECK-LABEL: # main

extern int printf(char* fmt, int a, int b);

int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int main() {
    for (int i = 0; i <= 10; i = i + 1) {
        printf("fib(%d) = %d\n", i, fib(i));
    }
    return 0;
}
