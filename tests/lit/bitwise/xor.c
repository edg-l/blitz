// RUN: %tinyc %s -o %t && %t
// EXIT: 6
int main() { return 12 ^ 10; }
