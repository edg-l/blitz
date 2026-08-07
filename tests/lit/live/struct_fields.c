// An array of four-field structs, reading three fields and writing a fourth,
// with the data seeded at runtime.
//
// This is `struct_walk` with the contents hidden from the compiler. A write to
// `recs[i].total` cannot reach `recs[i].a`, `.b` or `.c`, so the offset-aware
// half of `alias.rs` decides whether those loads survive the pass.

// OUTPUT: 217736
// EXIT: 0

extern int printf(char* fmt, int x);

struct Rec {
    int a;
    int b;
    int c;
    int total;
};

int main(int argc, char** argv) {
    int chk0 = 0;
    int reps = 423 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        struct Rec recs[128];
        int seed = ((argc + r) * 11) & 31;

        for (int i = 0; i < 128; i = i + 1) {
            recs[i].a = (i * 3 + seed) & 255;
            recs[i].b = (i * 5 + 1) & 255;
            recs[i].c = (i + seed) % 11;
            recs[i].total = 0;
        }

        for (int pass = 0; pass < 8; pass = pass + 1) {
            for (int i = 0; i < 128; i = i + 1) {
                recs[i].total = recs[i].total + recs[i].a;
                recs[i].total = recs[i].total + recs[i].b;
                recs[i].total = recs[i].total + recs[i].c;
            }
        }

        int sum = 0;
        for (int i = 0; i < 128; i = i + 1) {
            sum = sum + recs[i].total;
        }
        chk0 = (chk0 + (sum)) & 1048575;
    }
    printf("%d\n", chk0);
    return 0;
}
