// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int add3(int a, int b, int c) { return a + b + c; }
int main() { return add3(10, 20, 12); }
