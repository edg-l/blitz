// EXIT: 0
// Forward function declarations -- regression test.
int bar(int x);
int foo(int x) {
    if (x <= 0) { return 0; }
    return bar(x - 1) + 1;
}
int bar(int x) {
    if (x <= 0) { return 0; }
    return foo(x - 1) + 2;
}
int main() {
    if (foo(4) != 6) { return 1; }
    return 0;
}
