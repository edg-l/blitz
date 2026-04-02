// RUN: %tinyc %s -o %t && %t
// EXIT: 10
// Chained inlining: dbl(inc(4)) = dbl(5) = 10, fully folded.
int inc(int x) { return x + 1; }
int dbl(int x) { return x + x; }
int main() { return dbl(inc(4)); }
