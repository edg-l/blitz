// OUTPUT: 5.500000
// OUTPUT: 1.500000
// OUTPUT: 7.000000
// OUTPUT: 1.750000

extern int printf(char* fmt, double x);
int main() {
    double a = 3.5;
    double b = 2.0;
    printf("%f\n", a + b);
    printf("%f\n", a - b);
    printf("%f\n", a * b);
    printf("%f\n", a / b);
    return 0;
}
