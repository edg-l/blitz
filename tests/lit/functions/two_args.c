// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int add(int a, int b) { return a + b; }
int main() { return add(20, 22); }
