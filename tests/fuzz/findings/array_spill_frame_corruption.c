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
// Printing the right number while returning the wrong exit code, then
// crashing as pressure grows, is the signature of spilled values landing on
// top of something else in the frame. compute_frame_layout documents that
// "user stack slots sit at higher offsets than regalloc spill slots", so the
// suspect is spill slot indices colliding with the user array's region, or a
// frame sized without room for both.
//
// The frame-layout property tests in src/x86/abi.rs cover the layout
// arithmetic but not the assignment of slot indices to it, which is why they
// pass while this fails.
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
