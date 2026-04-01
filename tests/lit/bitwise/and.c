// RUN: %tinyc %s -o %t && %t
// EXIT: 8
int main() { return 12 & 10; }
