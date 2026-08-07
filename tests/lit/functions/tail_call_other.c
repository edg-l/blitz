// A tail call to a *different* function is the frame coming down and a jump.
//
// The counterpart to `tail_self_call.c`, and the two differ in one thing that
// decides everything else: whether the frame this function built is the one the
// callee will run in. A self-call jumps to the label after the prologue and the
// frame stands. Here the callee builds its own, so this one is torn down first --
// `setup_call_args`, then `abi::emit_frame_teardown`, then `jmp <symbol>`. RSP is
// then back on the return address the original `call` pushed, so `pong`'s `ret`
// returns straight past `ping` to `main`.
//
// The teardown is the epilogue *without* its `ret`, which is the part that has to
// be got right: emitting the full epilogue returns before the jump ever runs, and
// the mutually recursive pair in `live/tail_recursion.c` printed 140106 instead of
// 264767 until it was split out.
//
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # ping
// CHECK-NOT: call
// CHECK-LABEL: # pong
// CHECK-NOT: call
// The trailing label bounds the CHECK-NOT above: `main`'s own `printf` is a call
// and is not in tail position, so without this the scan runs into it.
// CHECK-LABEL: # main
// CHECK: call

extern int printf(char* fmt, ...);

int pong(int n, int acc);

__attribute__((noinline))
int ping(int n, int acc) {
    if (n <= 0) {
        return acc;
    }
    return pong(n - 1, acc + n);
}

__attribute__((noinline))
int pong(int n, int acc) {
    if (n <= 0) {
        return acc;
    }
    return ping(n - 1, acc + n * 2);
}

int main(int argc, char** argv) {
    printf("%d\n", ping(60 * argc, 0));
    return 0;
}

// OUTPUT: 2730
// EXIT: 0
