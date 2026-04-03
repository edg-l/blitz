// OUTPUT: sum: 5.750000
// OUTPUT: product: 17.500000
// OUTPUT: mixed: 15.700000

extern int printf(char* fmt, double x);

double add_doubles(double a, double b) {
    return a + b;
}

double multiply(double a, double b) {
    return a * b;
}

double mixed_args(int n, double x) {
    return n + x;
}

int main() {
    double r1 = add_doubles(2.5, 3.25);
    printf("sum: %f\n", r1);

    double r2 = multiply(3.5, 5.0);
    printf("product: %f\n", r2);

    double r3 = mixed_args(10, 5.7);
    printf("mixed: %f\n", r3);

    return 0;
}
