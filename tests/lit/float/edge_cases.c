// OUTPUT: neg: -3.500000
// OUTPUT: large: 1000000.000000
// OUTPUT: small: 0.000001
// OUTPUT: div_large: 0.500000

extern int printf(char* fmt, double x);

int main() {
    double a = 3.5;
    double neg = 0.0 - a;
    printf("neg: %f\n", neg);

    double big = 1000000.0;
    printf("large: %f\n", big);

    double tiny = 0.000001;
    printf("small: %f\n", tiny);

    double x = 50.0;
    double y = 100.0;
    printf("div_large: %f\n", x / y);

    return 0;
}
