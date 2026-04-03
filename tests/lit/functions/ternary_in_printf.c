// EXIT: 0
// OUTPUT: abs(-5) = 5
// OUTPUT: abs(3) = 3
// OUTPUT: abs(0) = 0

extern int printf(char* fmt, int a, int b);

int my_abs(int x) {
    return x > 0 ? x : 0 - x;
}

int main() {
    printf("abs(%d) = %d\n", 0 - 5, my_abs(0 - 5));
    printf("abs(%d) = %d\n", 3, my_abs(3));
    printf("abs(%d) = %d\n", 0, my_abs(0));
    return 0;
}
