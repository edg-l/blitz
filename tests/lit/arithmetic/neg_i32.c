// RUN: %tinyc %s -o %t && %t
// EXIT: 5
int main() { int x = -5; return -x; }
