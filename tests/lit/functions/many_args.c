// EXIT: 0
// OUTPUT: sum = 21

// 6 args uses all integer arg registers (rdi, rsi, rdx, rcx, r8, r9)
extern int printf(char* fmt, int a);

int sum6(int a, int b, int c, int d, int e, int f) {
    return a + b + c + d + e + f;
}

int main() {
    int result = sum6(1, 2, 3, 4, 5, 6);
    printf("sum = %d\n", result);
    return 0;
}
