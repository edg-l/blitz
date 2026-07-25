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
// ── Diagnosis (from --emit-asm, not the frame layout) ───────────────────────
//
// Not a frame-layout bug: the layout and slot indices are fine. The store of
// arr[6] is emitted with the array base as its own index register:
//
//   mov  r13, 0x18            ; byte offset of arr[6]
//   lea  r14,[rax+r13*1]      ; r14 = &arr[6]      (rax = array base)
//   mov  [rsp], r14           ; the address is SPILLED
//   mov  r13d, 0xb            ; r13 reused for the value 11
//   mov  DWORD PTR [rax+rax*1], r13d   ; <-- index is rax, not the offset
//
// The address vreg was spilled, so the correct lowering reloads it. Instead
// `build_mem_addr` (src/compile/effectful.rs) folded the Addr node and looked
// its children's registers up fresh, and by then the offset register had been
// reused for the stored value, yielding [rax+rax*1] -- a store far outside the
// frame, which is what clobbers the return path and then the return address.
//
// build_mem_addr already intends to guard this: "If the addr VReg came from a
// SpillLoad or cross-block import, the children's registers aren't guaranteed
// live at the load/store point." The guard tests whether an Op::Addr
// instruction defines the addr vreg, which is still true after the value has
// been spilled, so it does not fire. The guard needs to ask whether the
// address is still in its original register at this point, not whether it was
// ever computed by an Addr.
//
// Resolving registers at the barrier's own program point instead of the block
// exit was tried and changes nothing, since the stale lookup is of the Addr
// node's children rather than of the address itself.
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
