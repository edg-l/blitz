// RUN: %tinyc %s -o %t && %t
// EXIT: 30
// Multiple calls to the same function, all inlined.
int square(int x) { return x * x; }
int main() {
    int a = square(3);
    int b = square(4);
    int c = square(1);
    return a + b + c + square(2);
}
