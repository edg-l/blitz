// OUTPUT: int_to_double: 42.000000
// OUTPUT: double_to_int: 3.000000
// OUTPUT: mixed_add: 15.000000
// OUTPUT: float_to_double: 1.250000

extern int printf(char* fmt, double x);

int main() {
    int i = 42;
    double d = i;
    printf("int_to_double: %f\n", d);

    double e = 3.7;
    int j = e;
    double j_as_double = j;
    printf("double_to_int: %f\n", j_as_double);

    int k = 5;
    double f = 10.0;
    double result = k + f;
    printf("mixed_add: %f\n", result);

    float m = 1.25f;
    double n = m;
    printf("float_to_double: %f\n", n);

    return 0;
}
