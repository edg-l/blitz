// RUN: %tinyc %s -o %t && %t
// EXIT: 5
// Inline a function that contains an if/else branch.
int abs_val(int x) {
    if (x < 0) { return 0 - x; }
    return x;
}
int main() {
    return abs_val(0 - 5) + abs_val(0);
}
