// Test: XMM block param live across a call inside a loop - explicit block-param split.
//
// An XMM block parameter (the loop accumulator `acc`) crosses a call on each
// iteration. The splitter detects the block-param victim and applies the
// SlotSpillBlockParam strategy: predecessor terminators emit slot stores instead
// of register phi copies, and uses in the block reload from the slot.
//
// This is a focused test for the Phase 6 SlotSpillBlockParam path, distinct from
// xmm_loop_crossing.c. Here we verify a loop accumulator that is itself a block
// param survives being slot-spilled across a call.
//
// Requires BLITZ_SPLIT=1.
//
// OUTPUT: 15.000000
// EXIT: 0

extern int printf(char* fmt, double x);

__attribute__((noinline))
double add_one(double x) {
    return x + 1.0;
}

int main() {
    double acc = 0.0;
    int i = 0;
    // Loop: acc = add_one(acc) 5 times => acc = 5.0
    // Then multiply by 3.0 => 15.0
    while (i < 5) {
        acc = add_one(acc);
        i = i + 1;
    }
    acc = acc * 3.0;
    printf("%f\n", acc);
    return 0;
}
