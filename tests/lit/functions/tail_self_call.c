// A tail call to the same function is a jump to the top of its own body.
//
// `sum_to` calls itself in tail position, so no `call` is emitted for it at all:
// the arguments go into their ABI registers exactly as a call needs them, and
// control goes to the label bound *after* the prologue. The frame is neither torn
// down nor rebuilt, RSP does not move, and the body starts by moving each
// parameter out of its argument register -- which is the state a fresh entry
// would be in.
//
// No `call` also means no return address pushed, so the base case's `ret` returns
// straight to the original caller. That is where most of the win is: a recursion
// deeper than the return-address predictor makes every `ret` mispredict, and this
// removes the pair. Measured on `tests/lit/live/tail_recursion.c`, -19.3% cycles
// for -0.8% instructions, which is why the ranking is cycles.
//
// `not_tail` is the control: `n * f(n - 1)` multiplies *after* the call returns,
// so the call is not in tail position and must survive.
//
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # sum_to
// CHECK-NOT: call
// CHECK-LABEL: # not_tail
// CHECK: call

extern int printf(char* fmt, ...);

int sum_to(int n, int acc) {
    if (n <= 0) {
        return acc;
    }
    return sum_to(n - 1, acc + n);
}

int not_tail(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * not_tail(n - 1);
}

int main(int argc, char** argv) {
    printf("%d\n", sum_to(100 * argc, 0));
    printf("%d\n", not_tail(5 * argc));
    return 0;
}

// OUTPUT: 5050
// OUTPUT: 120
// EXIT: 0
