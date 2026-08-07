// A hot loop over a callee the inliner may not touch.
//
// Every iteration pays the ABI: arguments into their registers, the call, the
// result out of RAX, and every caller-saved value spilled around it. That cost
// is invisible in a corpus where everything inlines, and it is what the
// inliner's own pressure check has to be measured against.

// OUTPUT: 565296
// EXIT: 0

extern int printf(char* fmt, ...);

__attribute__((noinline))
int mix(int a, int b, int c) {
    int t = (a * 3 + b) & 1023;
    if (t > c) {
        t = t - c;
    } else {
        t = t + c;
    }
    return t & 511;
}

int main(int argc, char** argv) {
    int chk0 = 0;
    int reps = 171 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 17) & 63;
        int acc = 0;
        for (int i = 0; i < 1024; i = i + 1) {
            acc = acc + mix(i & 255, seed, (i * 7) & 255);
            acc = acc & 65535;
        }
        chk0 = (chk0 + (acc)) & 1048575;
    }
    printf("%d\n", chk0);
    return 0;
}
