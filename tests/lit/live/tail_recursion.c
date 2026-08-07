// Tail recursion, which is a loop written the other way round.
//
// `step` calls itself in tail position and nowhere else, so every call is a
// frame blitz builds and tears down for an iteration that needs neither.
// `gcc -O2` turns it into a jump and the recursion disappears; blitz does not
// have tail-call optimization, and this kernel is what would price it.
//
// The corpus had nothing to price it with: `bench`, `live` and the generated
// programs contain **zero** tail-call sites between them, and the 59 in `lit`
// are almost all `main` returning a call once. A transform with no kernel is a
// transform whose value is asserted rather than measured, which is how the
// inliner's pressure check stayed open for three sessions.
//
// The accumulator makes the recursion genuinely tail-shaped rather than
// `n * f(n-1)`, which is not a tail call and would measure something else.

// OUTPUT: 990778
// OUTPUT: 264767
// EXIT: 0

extern int printf(char* fmt, ...);

int step(int n, int acc) {
    if (n <= 0) {
        return acc & 65535;
    }
    return step(n - 1, (acc + n * 3) & 65535);
}

// Two mutually tail-recursive functions: the same shape across a call edge the
// inliner will not collapse, which is where a general tail call earns more than
// a self-call rewritten as a loop.
int odd_step(int n, int acc);

int even_step(int n, int acc) {
    if (n <= 0) {
        return acc & 65535;
    }
    return odd_step(n - 1, (acc + n) & 65535);
}

int odd_step(int n, int acc) {
    if (n <= 0) {
        return acc & 65535;
    }
    return even_step(n - 1, (acc + n * 2) & 65535);
}

int main(int argc, char** argv) {
    int chk0 = 0;
    int chk1 = 0;
    int reps = 1100 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 13) & 255;
        chk0 = (chk0 + step(seed, r & 255)) & 1048575;
        chk1 = (chk1 + even_step(seed, r & 255)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    return 0;
}
