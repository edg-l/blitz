// RUN: %tinyc %s -o %t && %t
// EXIT: 0
int zero() { return 0; }
int main() { return zero(); }
