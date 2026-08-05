// A call argument's ordering constraint must reach the VReg the block itself
// defines for that class.
//
// `keys` is passed to `probe` from three separate loops, so linearization
// re-emits its `StackAddr` in each of those blocks: one e-class, several VRegs.
// `build_barrier_maps` resolved the argument class to a VReg by asking the
// function-wide map at *block entry*, which answered with the copy some earlier
// block had emitted. The block's own copy therefore carried no constraint, the
// scheduler was free to place its definition after the call that reads it, and
// the emitted code passed whatever the register held on entry:
//
//   mov  rdi,r15          ; pass keys
//   call probe
//   lea  r15,[rsp+0x40]   ; ... and define it here, after the call
//
// R15 was zero on the first pass, so `probe` dereferenced a null pointer.
// `BLITZ_VERIFY=1` named it exactly -- "reads R15 on a path where nothing
// writes it" -- while the ordinary run segfaulted at -O0 and was correct at -O1.

// OUTPUT: 210
// OUTPUT: 20
// EXIT: 0

extern int printf(char* fmt, int x);

int probe(int* keys, int mask, int key) {
    unsigned int h = ((unsigned int)key * 2654435761) & (unsigned int)mask;
    while (keys[(int)h] != 0 && keys[(int)h] != key) {
        h = (h + 1) & (unsigned int)mask;
    }
    return (int)h;
}

int main() {
    int keys[64];
    int vals[64];
    for (int i = 0; i < 64; i = i + 1) {
        keys[i] = 0;
        vals[i] = 0;
    }

    for (int i = 1; i <= 20; i = i + 1) {
        int slot = probe(keys, 63, i * 7 + 1);
        keys[slot] = i * 7 + 1;
        vals[slot] = i;
    }

    int hits = 0;
    for (int i = 1; i <= 20; i = i + 1) {
        int slot = probe(keys, 63, i * 7 + 1);
        if (keys[slot] == i * 7 + 1) {
            hits = hits + vals[slot];
        }
    }

    int absent = 0;
    for (int i = 1; i <= 20; i = i + 1) {
        int slot = probe(keys, 63, i * 7 + 4);
        if (keys[slot] == 0) {
            absent = absent + 1;
        }
    }

    printf("%d\n", hits);
    printf("%d\n", absent);
    return 0;
}
