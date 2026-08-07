// Double-precision dot product and a scaled accumulate over 256 elements.
//
// The XMM path with a loop-carried accumulator: exactly one value must survive
// every iteration in a register class where every register is caller-saved.

// Results are reported as scaled integers, never as doubles: a `printf`
// prototype that names a `double` parameter is not variadic, so a compiler that
// believes it leaves AL unset and the library prints nothing useful. Blitz sets
// AL on every call and would print the right answer, which makes the reference
// compiler the one that looks wrong.

// OUTPUT: 60408
// OUTPUT: 5985
// EXIT: 0

extern int printf(char* fmt, ...);

int main() {
    double x[256];
    double y[256];

    for (int i = 0; i < 256; i = i + 1) {
        x[i] = (double)(i % 17) * 0.5;
        y[i] = (double)(i % 23) * 0.25 + 1.0;
    }

    double dot = 0.0;
    for (int i = 0; i < 256; i = i + 1) {
        dot = dot + x[i] * y[i];
    }

    double scaled = 0.0;
    for (int i = 0; i < 256; i = i + 1) {
        scaled = scaled + (x[i] * 2.0 + y[i]) * 0.125;
    }

    // Both sums are exact in binary floating point (every term is a multiple of
    // 1/16), so the scaled integers are exact too.
    printf("%d\n", (int)(dot * 16.0));
    printf("%d\n", (int)(scaled * 16.0));
    return 0;
}
