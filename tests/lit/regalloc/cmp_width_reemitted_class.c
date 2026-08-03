// Regression test: a flags-only 32-bit compare emitted at 64-bit width, so a
// negative value compared as a large positive one and the branch inverted.
//
//   cmp r8,rdi     <-- what blitz emitted
//   jl  ...
//
// `mov edi,0xfffffffe` zero-extends, so RDI holds 4294967294 and `14 < -2` came
// out true. `14 <= -2` too; `-2 > 14`, and a compare against a literal, are all
// fine, which is what makes this narrow.
//
// The width came from `vreg_types`, missing for this VReg. A class re-emitted in a
// later block gets a VReg of its own, and the restore after each block in
// linearization is an `insert_single` -- which replaces the class's segments, so
// the function-wide map keeps one re-emission and the others have no type at all.
// `vreg_types` is now built from the per-block snapshots as well, where every
// re-emission is recorded. The 64-bit fallback that turned a missing entry into
// wrong code is still there; what is fixed is the entry being missing.
//
// The shape is specific because it needs the compare to reach the dead-difference
// path in lowering while the same subtraction is also live: `apply_icmp_isel`
// rewrites `Icmp(cc, a, b)` to `Proj1(X86Sub(a, b))` and one X86Sub is shared, so
// the class is emitted once for its difference (with a type, 32-bit) and once for
// its flags in another block (without one, 64-bit).
//
// From gen_c.py seed 58 shape mixed, 666 lines, reduced to 13 by hand: cc 1842
// against blitz 1850 at BOTH levels, which no self-consistency oracle can see.
// One term was wrong -- `arr[4]` read 81 where its initialiser 5 belonged, from
// `if ((v2 < v6)) { arr[((58 - v2)) & 7] = 81; }` running when it must not, and
// `(58 - 14) & 7` is 4.
//
// EXIT: 0
// OUTPUT: 5
extern int printf(char* fmt, int x);
int main() {
    int x = 5;
    int v2 = 14;
    int v6 = -2;
    for (int i = 0; i < 1; i++) {
        if ((i == (v2 - v6))) {
        } else {
            if ((v2 < v6)) { x = 81; }
        }
    }
    printf("%d\n", x);
    return 0;
}
