// RUN: %tinyc %s -o %t && %t
// EXIT: 15
int inner(int x) { return x + 5; }
int middle(int x) { return inner(x) + 3; }
int main() { return middle(7); }
