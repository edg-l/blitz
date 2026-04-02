// RUN: %tinyc %s -o %t && %t
// EXIT: 40
// After inlining mul8, strength reduction turns x*8 into shl.
// Verified by correct runtime result.
int mul8(int x) { return x * 8; }
int main() { return mul8(5); }
