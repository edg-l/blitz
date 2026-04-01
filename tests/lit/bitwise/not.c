// RUN: %tinyc %s -o %t && %t
// EXIT: 245
int main() { return ~10 & 255; }
