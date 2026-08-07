// Eight bodies, ten velocity-and-position update steps, all double.
//
// The inner pair loop keeps six coordinates and three deltas live at once, so
// the XMM class is under real pressure while a call-free loop runs. Division is
// the expensive op the cost model has to price correctly.

// Reported as scaled integers rather than doubles: see the note in
// `dot_product.c` -- a non-variadic `printf` prototype leaves AL unset on a
// compiler that believes it, and the reference is the one that then prints
// nothing useful.

// OUTPUT: 36000
// OUTPUT: 36000
// EXIT: 0

extern int printf(char* fmt, ...);

int main() {
    double px[8];
    double py[8];
    double vx[8];
    double vy[8];
    double mass[8];

    for (int i = 0; i < 8; i = i + 1) {
        px[i] = (double)(i + 1);
        py[i] = (double)(8 - i);
        vx[i] = 0.0;
        vy[i] = 0.0;
        mass[i] = 1.0 + (double)(i % 3) * 0.5;
    }

    for (int step = 0; step < 10; step = step + 1) {
        for (int i = 0; i < 8; i = i + 1) {
            double ax = 0.0;
            double ay = 0.0;
            for (int j = 0; j < 8; j = j + 1) {
                if (i != j) {
                    double dx = px[j] - px[i];
                    double dy = py[j] - py[i];
                    // Softened so the denominator can never be zero.
                    double d2 = dx * dx + dy * dy + 0.5;
                    double inv = mass[j] / (d2 * d2);
                    ax = ax + dx * inv;
                    ay = ay + dy * inv;
                }
            }
            vx[i] = vx[i] + ax * 0.01;
            vy[i] = vy[i] + ay * 0.01;
        }
        for (int i = 0; i < 8; i = i + 1) {
            px[i] = px[i] + vx[i] * 0.01;
            py[i] = py[i] + vy[i] * 0.01;
        }
    }

    double sx = 0.0;
    double sy = 0.0;
    for (int i = 0; i < 8; i = i + 1) {
        sx = sx + px[i];
        sy = sy + py[i];
    }
    // Rounded to 1/1000, well above the last-bit differences a different
    // instruction order can produce.
    printf("%d\n", (int)(sx * 1000.0 + 0.5));
    printf("%d\n", (int)(sy * 1000.0 + 0.5));
    return 0;
}
