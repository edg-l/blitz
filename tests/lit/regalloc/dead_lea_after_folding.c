// The address computation a load folded into its own addressing mode is not
// emitted twice.
//
// Lowering puts `[base + idx*4]` into the addressing mode of the load that uses
// it, and it decides that per consumer at the last possible moment. When every
// consumer folds, the `lea` that computed the address is left with nothing
// reading it -- and no earlier pass can see that, because DCE runs on the CFG
// before scheduling and the e-graph never sees an effectful op. Measured over
// `bench` and `live` before `emit::dead_inst` existed: 122 of 7801 instructions,
// in every one of the 34 kernels, all of them inside loop bodies.
//
// The loop below is the shape that produces one: an array indexed by a variable,
// where the index has to be sign-extended and scaled. `i` comes from `argc` so
// nothing folds at compile time.
//
// Behavioural, deliberately, with no `lea` count. What the pass *removes* is
// already pinned by `tests/baselines/codesize-*.tsv`, which track `insts` and
// `copies` per program and moved on 44 `lit` rows when it landed; a count here
// would pin a number the allocator is entitled to change and would say nothing
// about correctness. What could go *wrong* -- deleting something live -- is what
// this file is for, and what `run_diff.sh` and the generated corpus check it
// against. The first version of the pass deleted every call's argument setup, and
// 248 lit tests said so.
//
// RUN: %tinyc %s -o %t && %t

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int arr[64];
    int i = 0;
    while (i < 64) {
        arr[i] = i * 3 + argc;
        i = i + 1;
    }
    int total = 0;
    int k = argc & 7;
    while (k < 64) {
        total = total + arr[k] + arr[k];
        k = k + 8;
    }
    printf("%d\n", total);
    return 0;
}

// OUTPUT: 1408
// EXIT: 0
