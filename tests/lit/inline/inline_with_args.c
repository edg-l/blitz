// RUN: %tinyc %s -o %t && %t
// EXIT: 30
int add(int a, int b) { return a + b; }
int main() { return add(10, 20); }
