// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Inlined function with multiple return paths, result used in comparison.

int clamp(int x, int lo, int hi) {
    if (x < lo) { return lo; }
    if (x > hi) { return hi; }
    return x;
}

__attribute__((noinline))
int identity(int x) { return x; }

int main() {
    int a = clamp(identity(50), 0, 100);
    int b = clamp(identity(0 - 5), 0, 100);
    int c = clamp(identity(200), 0, 100);
    if (a != 50) { return 1; }
    if (b != 0) { return 2; }
    if (c != 100) { return 3; }
    return 0;
}
