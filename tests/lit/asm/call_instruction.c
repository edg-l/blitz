// RUN: %tinyc %s --emit-asm 2>&1
// CHECK: call
//
// `+ 1` is what keeps this a call. `return foo(42)` is a tail call, which
// lowering turns into a jump to `foo` -- correct, and no longer a test of the
// call instruction. `lit/functions/tail_self_call.c` covers that transform.
__attribute__((noinline))
int foo(int x) { return x; }
int main() { return foo(42) + 1 - 1; }
