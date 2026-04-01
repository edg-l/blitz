// RUN: %tinyc %s -o %t && %t
// EXIT: 5
int main() { return (-15) / 3 + 10; }
