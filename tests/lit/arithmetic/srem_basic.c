// RUN: %tinyc %s -o %t && %t
// EXIT: 2
int main() { return 17 % 5; }
