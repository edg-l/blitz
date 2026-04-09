// EXIT: 0
// Test new operators working together in realistic patterns.

__attribute__((noinline)) int count_bits(int n) {
    int count = 0;
    while (n > 0) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

int main() {
    // Bit counting with >>= and +=
    if (count_bits(0) != 0) { return 1; }
    if (count_bits(1) != 1) { return 2; }
    if (count_bits(7) != 3) { return 3; }
    if (count_bits(0xFF) != 8) { return 4; }

    // Fibonacci with compound operators
    int fib_prev = 0;
    int fib_curr = 1;
    for (int i = 0; i < 10; i++) {
        int next = fib_prev + fib_curr;
        fib_prev = fib_curr;
        fib_curr = next;
    }
    if (fib_curr != 89) { return 5; }

    return 0;
}
