// RUN: %tinyc %s -o %t && %t
// EXIT: 20
// Chain of 3 inlined functions: inc -> double -> add_ten.
// All constant args, should fold to 20.
int inc(int x) { return x + 1; }
int double_it(int x) { return x + x; }
int add_ten(int x) { return x + 10; }
int main() { return add_ten(double_it(inc(4))); }
