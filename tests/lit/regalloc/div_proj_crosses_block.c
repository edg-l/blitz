// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// OUTPUT: 5
//
// `v % 8` and `v / 8` are one x86_idiv node, and the two projections that read
// its results are consumed in different blocks. A division leaves its quotient
// in RAX and its remainder in RDX, which no block boundary preserves, so the
// block taking the quotient has to run the division itself rather than project
// the pair the outer block emitted -- projecting it there reads whichever
// register that block last wrote, never the quotient.

extern int printf(char* fmt, int x);

__attribute__((noinline))
int id(int x) { return x; }

int main() {
    int v = id(-90);
    int q = 0;
    int r = 0;
    if (v < 0) {
        r = v % 8;
        if (r < 0) {
            q = v / 8;
        }
    }
    printf("%d\n", (q & 7));
    return 0;
}
