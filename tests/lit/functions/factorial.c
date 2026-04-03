// EXIT: 0
// OUTPUT: 0! = 1
// OUTPUT: 1! = 1
// OUTPUT: 2! = 2
// OUTPUT: 3! = 6
// OUTPUT: 4! = 24
// OUTPUT: 5! = 120
// OUTPUT: 6! = 720
// OUTPUT: 7! = 5040
// OUTPUT: 8! = 40320
// OUTPUT: 9! = 362880
// OUTPUT: 10! = 3628800
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # factorial
// recursive call
// CHECK: call
// CHECK-LABEL: # main

extern int printf(char* fmt, int a, int b);

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int main() {
    for (int i = 0; i <= 10; i = i + 1) {
        printf("%d! = %d\n", i, factorial(i));
    }
    return 0;
}
