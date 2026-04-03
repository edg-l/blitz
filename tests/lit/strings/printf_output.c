// EXIT: 0
// OUTPUT: x = 10
// OUTPUT: y = 20
// OUTPUT: sum = 30

extern int printf(char* fmt, int x);
int main() {
    int x = 10;
    int y = 20;
    printf("x = %d\n", x);
    printf("y = %d\n", y);
    printf("sum = %d\n", x + y);
    return 0;
}
