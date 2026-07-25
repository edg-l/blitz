// KNOWN FAILING -- reduced from tests/fuzz/findings/seed5_miscompile.c.
// Do not "fix" by weakening it.
//
// Summing enough elements of a stack array corrupts the frame. The value is
// still right until it isn't, and the damage escalates with the number of
// simultaneously live loads:
//
//   elements   blitz                      gcc
//   <= 5       correct, exit 0            same
//   6          correct value, exit 2      exit 0     <- return path clobbered
//   >= 7       SIGSEGV (exit 139)         correct    <- return address clobbered
//
// ── Fixed ───────────────────────────────────────────────────────────────────
//
// Four defects compounded here, all fixed (8acd79b, 350d36a, and the reload
// ordering below). This file now matches gcc and is a live regression test.
//
//   1. build_mem_addr folded an addressing mode using registers re-resolved
//      through the class map, picking a VReg the LEA had not used.
//   2. A load or store whose address had been spilled resolved to the
//      pre-spill register instead of the reload.
//   3. `return 0` after a call emitted nothing, because the constant resolved
//      to RAX -- which held the callee's return value.
//   4. A reload could be emitted in group 0, ahead of the store to its slot,
//      so it loaded whatever the caller had left on the stack. Reloads are now
//      pinned after the store to their slot.
//
// OUTPUT: 28
// EXIT: 0

extern int printf(char* fmt, int x);

int main() {
    int arr[8];
    arr[0] = -7;
    arr[1] = -4;
    arr[2] = -1;
    arr[3] = 2;
    arr[4] = 5;
    arr[5] = 8;
    arr[6] = 11;
    arr[7] = 14;
    printf("%d\n",
           arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]);
    return 0;
}
