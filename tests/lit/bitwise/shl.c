// RUN: %tinyc %s -o %t && %t
// EXIT: 40
int main() { return 5 << 3; }
