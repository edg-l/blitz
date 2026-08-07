// EFLAGS is shared between comparisons and destroyed by the calls between them.
//
// `apply_icmp_isel` merges every `Icmp(cc, a, b)` over one pair of operands
// onto a single shared compare, deliberately, so one `cmp` serves several
// conditions. That is sound as far as it goes -- `a == b` and `a != b` really
// do set identical flags -- but it makes one flags value live across everything
// between its uses, and a `call` clobbers EFLAGS.
//
// The emitted code is one compare and four correct `cmov`s, with a `printf`
// between each pair:
//
//     c:  cmp    eax,0x1
//    3b:  cmove  edx,eax        <- reads the compare's flags
//    6d:  cmovne edx,eax        <- reads printf's flags
//    9f:  cmovl  edx,eax        <- reads printf's flags
//    d1:  cmovge edx,eax        <- reads printf's flags
//
// A flags value cannot be spilled and reloaded -- no store reaches EFLAGS -- so
// it is rematerialized instead, the way a constant is: `compile::flags_remat`
// re-emits the comparison wherever something has written flags since it was
// computed. It runs before register allocation, so the operands the re-emitted
// compare reads have their live ranges extended in the graph the allocator
// colours. `regalloc::fast` stated the assumption that broke -- "nothing this
// pass inserts between a comparison and its consumer writes EFLAGS: a spill
// load and a spill store are both `mov`" -- which covers spills and said
// nothing about calls.
//
// WRITING THE COMPARISONS AWAY FROM THE CALLS DOES NOT HELP: computing all
// four into locals first and printing them afterwards emits the same thing,
// because the scheduler sinks each `cmov` to its use. The interleaving is the
// compiler's, not the source's.
//
// BRANCHES ARE UNAFFECTED: a branch takes its cc from the CFG's own `Branch`
// op and the terminator sits at the end of its block, so no call intervenes.
// That is why `gen_c.py` and the lit corpus, which compare inside `if`
// conditions, never reach this.
//
// A SEPARATE BUG THAT USED TO HIDE THIS ONE IS FIXED: the cc was recovered from
// the flags class by `find_cc_in_class`, which returned whichever `Icmp` node
// came first, so all four comparisons above lowered to the same condition. The
// cc now rides on `PureOp::Select`, for the same reason it rides on
// `EffectfulOp::Branch`.
//
// OUTPUT: 1
// OUTPUT: 0
// OUTPUT: 0
// OUTPUT: 1
// OUTPUT: 0
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int a = argc;
    int b = 1;
    printf("%d\n", a == b);
    printf("%d\n", a != b);
    printf("%d\n", a < b);
    printf("%d\n", a >= b);
    printf("%d\n", !a);
    return 0;
}
