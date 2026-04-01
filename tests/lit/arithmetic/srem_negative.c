// RUN: %tinyc %s -o %t && %t
// EXIT: 7
int main() { return (-17) % 5 + 9; }
