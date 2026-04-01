// RUN: %tinyc %s -o %t && %t
// EXIT: 5
int main() { return 40 >> 3; }
