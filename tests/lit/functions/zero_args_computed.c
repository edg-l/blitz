// EXIT: 0
// Zero-arg function returning a computed value.
int compute() {
    int a = 3;
    int b = 4;
    return a * a + b * b;
}
int main() {
    if (compute() != 25) { return 1; }
    return 0;
}
