// KNOWN FAILING -- reproducer for a real bug. Do not "fix" by weakening it.
//
//   cc -O0          0
//   blitz -O1       0
//   blitz -O0       1      <- the guard sees the wrong sum
//
// Hand-reduced from tests/fuzz/gen_c.py seed 6. Pre-existing: it fails
// identically at a6a4494, before any of this session's fixes.
//
// `e` is parameter 4, so the ABI puts it in XMM3, and it stays live until the
// last add of the guard. The intermediate `x + y` computed in the merge block
// is given XMM3 as well and overwrites it three instructions early:
//
//   3c:  movsd xmm3,xmm4     ; x + y  -- clobbers e
//   40:  addsd xmm3,xmm1     ;   + y
//   44:  movsd xmm0,xmm3
//   48:  addsd xmm0,xmm2     ;   + z
//   4c:  movsd xmm1,xmm0
//   50:  addsd xmm1,xmm3     ;   + w, reads (x+y) instead of e
//
// So the interference between a value live *through* both arms of the branch
// and one defined in the merge block is being missed. Both arms end in a jump
// to the merge block, and `e` is read only in the merge block, so it is
// live-in there and must interfere with anything defined before that read.
//
// Where to start: the interference graph for the merge block, and whether
// global_liveness marks `e`'s VReg live-in to it. Note the read of `e` in the
// merge block resolves through the block's class map, so check first that the
// VReg the merge block reads is the same one the entry block's Param defines
// -- the two neighbouring bugs fixed this session (438bdc4, ccc64b7) were both
// a block resolving a class to the wrong VReg.
//
// EXIT: 0

int pick(int c, double a, double b, double d, double e) {
    double x = a;
    double y = b;
    double z = d;
    double w = e;
    if (c > 0) {
        x = a + 1.0;
    } else {
        y = b + 1.0;
    }
    if (x + y + z + w != 100.0) { return 1; }
    return 0;
}

int main() {
    // else path: 30 + (26+1) + 12 + 31 = 100
    return pick(0, 30.0, 26.0, 12.0, 31.0);
}
