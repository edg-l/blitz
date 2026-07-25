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
// ── Status ─────────────────────────────────────────────────────────────────
//
// Three of the four defects behind this are fixed (8acd79b, 350d36a): the
// addressing-mode fold reading a stale index register, load/store addresses
// resolving to the pre-spill register, and `return 0` being dropped after a
// call. Five and six elements now match gcc exactly.
//
// Seven or more still segfaults, for a remaining splitter defect. The address
// of arr[6] is spilled:
//
//   lea  r14,[rax+r13*1]     ; r14 = &arr[6]
//   mov  [rsp], r14          ; spilled
//   mov  DWORD PTR [r15], r13d   ; <-- r15 never written
//
// The splitter rewrote the store barrier's operand from VReg(19) to a reload
// VReg it created, but never emitted the reload instruction ahead of the use:
// the barrier's operands are [VReg(1), VReg(90), VReg(111), VReg(20)] and
// VReg(19) is absent, while nothing loads [rsp] back before the store. The
// reload is inserted for a later consumer instead. So the plan is right and
// the insertion point is wrong -- a bug in where the splitter places reloads
// for effectful-op operands, not in how lowering resolves them.
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
