// RUN: %tinyc %s -o %t && %t
// EXIT: 6
// Inline a function with multiple return paths.
int pick(int x) {
    if (x == 1) { return 10; }
    if (x == 2) { return 20; }
    return 30;
}
int main() {
    int a = pick(1);
    int b = pick(2);
    int c = pick(99);
    return (a + b + c) - 54;
}
