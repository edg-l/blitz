// An ABI register is where a value has to be at a call, not a property it
// carries everywhere.
//
// `d2` is the first call's result and so is coloured XMM0; `-23.0` is the first
// floating-point argument of that same call and is coloured XMM0 too. The
// `-O0` allocator honoured both colours wherever it loaded either value, so
// `d0 + (d2 + d3)` loaded both operands of the inner add into XMM0 and added
// `-23.0` to itself. One term of the final sum, and the program exits 3 rather
// than 0.
//
// 36 lines, reduced from gen_c.py seed 180 shape mixed.
//
// EXIT: 0
double f2(double p0, int p1, double p2, double p3, double p4, double p5, int p6, double p7) {
    return ((p5 + p7) - 4.0);
}
int main() {
    int v0 = -12;
    int v1 = -28;
    int v4 = 28;
    int v5 = -17;
    int v6 = -47;
    int v8 = -50;
    int v9 = 45;
    double d0 = 23.0;
    double d1 = -15.0;
    double d2 = 28.0;
    double d3 = -23.0;
    double d4 = -11.0;
    double d5 = -23.0;
    v0 = (((10 - ((v6 & 1023) ^ 709)) & 1023) & 340);
    d2 = f2(d5, 97, d4, (d1 * d2), (d0 * d0), -34.0, (v6 % -6), (d1 + d3));
    d0 = (d0 + (d2 + d3));
    d2 = (d2 + (v0 & 63));
    d3 = (d3 + (-36.0 - d5));
    d5 = (d5 + ((42 * v4) & 63));
    if ((v1 <= v9)) {
        for (int i6 = 0; i6 < 1; i6++) {
        }
    }
    v5 = v4;
    d4 = f2((d0 - d3), ((v9 & 1023) & 260), d3, d5, d4, (d5 - d3), (-78 * v9), d3);
    d0 = (d0 + d3);
    d2 = (d2 + (d3 - -14.0));
    d3 = (d3 + (d3 * d2));
    d4 = (d4 + (d4 - d1));
    d5 = (d5 + ((v5 * v8) & 63));
    if ((((((d0 + d1) + d2) + d3) + d4) + d5) != 2585.0) { return 3; }
}
