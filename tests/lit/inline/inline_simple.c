// RUN: %tinyc %s -o %t && %t
// EXIT: 42
int leaf() { return 42; }
int main() { return leaf(); }
