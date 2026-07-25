// Regression test: a block param is live from block entry, not from its pseudo-op.
//
// A `BlockParam` computes nothing -- the value is already in its register when
// the block starts, put there by a predecessor's phi copy. Its pseudo-op was
// left wherever scheduling put it, though, and liveness reads a def position as
// the start of a live range. The backward pass that pulls definitions closer to
// their consumers moved param 4's pseudo-op down next to its use, so it looked
// dead over the earlier part of its own block and the allocator gave XMM3, its
// register, to the intermediate computed there:
//
//   movsd xmm3,xmm4     ; x + y, into the register holding param 4
//   addsd xmm3,xmm1
//   movsd xmm0,xmm3
//   addsd xmm0,xmm2
//   movsd xmm1,xmm0
//   addsd xmm1,xmm3     ; wanted param 4, reads x + y
//
// So the sum came out wrong at -O0 and right at -O1, which is why the
// O0-vs-O1 differential harness is what caught it. Hand-reduced from
// tests/fuzz/gen_c.py seed 6.
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
