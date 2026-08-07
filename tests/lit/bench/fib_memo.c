// Fibonacci three ways: recursive, iterative, and memoized through an array
// parameter.
//
// The recursive one is a call-heavy function whose argument and partial result
// both have to survive a call; the memoized one adds a load-guarded early
// return, so the two calls are conditional and most invocations return without
// one.

// OUTPUT: 1
// OUTPUT: 63245986
// OUTPUT: 63245986
// EXIT: 0


extern int printf(char* fmt, ...);

int fib_rec(int n) {
    if (n <= 1) {
        return n;
    }
    return fib_rec(n - 1) + fib_rec(n - 2);
}

int fib_iter(int n) {
    int a = 0;
    int b = 1;
    for (int i = 0; i < n; i = i + 1) {
        int t = a + b;
        a = b;
        b = t;
    }
    return a;
}

int fib_memo(int n, int* memo) {
    if (n <= 1) {
        return n;
    }
    if (memo[n] != 0) {
        return memo[n];
    }
    int v = fib_memo(n - 1, memo) + fib_memo(n - 2, memo);
    memo[n] = v;
    return v;
}

int main() {
    int memo[40];
    for (int i = 0; i < 40; i = i + 1) {
        memo[i] = 0;
    }

    int agree = 1;
    for (int n = 0; n <= 22; n = n + 1) {
        int a = fib_rec(n);
        int b = fib_iter(n);
        int c = fib_memo(n, memo);
        if (a != b || b != c) {
            agree = 0;
        }
    }

    printf("%d\n", agree);
    printf("%d\n", fib_iter(39));
    printf("%d\n", fib_memo(39, memo));
    return 0;
}
