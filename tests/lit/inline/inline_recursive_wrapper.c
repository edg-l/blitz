// RUN: %tinyc %s --emit-ir 2>&1
// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined wrapper around a noinline recursive function.
// fib() wrapper should be inlined: main calls fib_impl directly.
// CHECK: function main
// CHECK: call fib_impl

__attribute__((noinline))
int fib_impl(int n) {
    if (n <= 1) { return n; }
    return fib_impl(n - 1) + fib_impl(n - 2);
}

int fib(int n) { return fib_impl(n); }

int main() {
    int a = fib(0);
    int b = fib(1);
    int c = fib(5);
    int d = fib(10);
    if (a != 0) { return 1; }
    if (b != 1) { return 2; }
    if (c != 5) { return 3; }
    if (d != 55) { return 4; }
    return 0;
}
