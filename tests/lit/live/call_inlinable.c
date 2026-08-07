// A hot loop over a callee the inliner is free to take.
//
// `call_hot.c` is the same shape with `__attribute__((noinline))` on the callee,
// and that is the whole difference: it measures what a call costs when the
// inliner cannot act, this measures what the inliner's decision is worth. Until
// this existed, `live` had no inlinable call site at all -- 20 of its 21 kernels
// are a single function and the 21st is deliberately `noinline` -- so the
// inliner's effect on the only metric the Goal ranks was zero by construction.
//
// The callee is small and the loop holds little else, so inlining should win
// here: the ABI cost per iteration is arguments into registers, the call, the
// result out of RAX, and every caller-saved value spilled around it, against a
// body of four arithmetic ops.

// OUTPUT: 565296
// EXIT: 0

extern int printf(char* fmt, ...);

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
