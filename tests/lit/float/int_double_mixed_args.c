// Regression: a function taking both an int and a double. The int goes in RDI
// and the double in XMM0, so this catches an operand-class mistake that a
// single-class signature would not.
//
// Same root cause as tests/lit/float/int_to_double_conversions.c: the int
// parameter feeding cvtsi2sd was given an XMM register, and the conversion
// read 64 bits of a 32-bit value.
//
// OUTPUT: mixed: 15.700000
// EXIT: 0

extern int printf(char* fmt, ...);

double mixed(int n, double x) {
    return n + x;
}

int main() {
    printf("mixed: %f\n", mixed(10, 5.7));
    return 0;
}
