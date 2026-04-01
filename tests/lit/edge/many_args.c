// EXIT: 0
// Test function with more arguments than registers (> 6 on SysV).
int sum7(int a, int b, int c, int d, int e, int f, int g) {
    return a + b + c + d + e + f + g;
}
int main() {
    if (sum7(1, 2, 3, 4, 5, 6, 7) != 28) { return 1; }
    return 0;
}
