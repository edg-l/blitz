// OUTPUT: eq: 0
// OUTPUT: ne: 1
// OUTPUT: lt: 1
// OUTPUT: gt: 0
// OUTPUT: le_eq: 1
// OUTPUT: ge_eq: 1

extern int printf(char* fmt, int x);

__attribute__((noinline))
int check_eq(double a, double b) {
    if (a == b) { return 1; }
    return 0;
}

__attribute__((noinline))
int check_ne(double a, double b) {
    if (a != b) { return 1; }
    return 0;
}

__attribute__((noinline))
int check_lt(double a, double b) {
    if (a < b) { return 1; }
    return 0;
}

__attribute__((noinline))
int check_gt(double a, double b) {
    if (a > b) { return 1; }
    return 0;
}

__attribute__((noinline))
int check_le(double a, double b) {
    if (a <= b) { return 1; }
    return 0;
}

__attribute__((noinline))
int check_ge(double a, double b) {
    if (a >= b) { return 1; }
    return 0;
}

int main() {
    printf("eq: %d\n", check_eq(2.5, 3.7));
    printf("ne: %d\n", check_ne(2.5, 3.7));
    printf("lt: %d\n", check_lt(2.5, 3.7));
    printf("gt: %d\n", check_gt(2.5, 3.7));
    printf("le_eq: %d\n", check_le(3.0, 3.0));
    printf("ge_eq: %d\n", check_ge(3.0, 3.0));
    return 0;
}
