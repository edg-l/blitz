// RUN: %tinyc %s -o %t && %t
// EXIT: 14
int main() { return 12 | 10; }
