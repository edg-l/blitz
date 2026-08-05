// A loop over an array of four-field structs, reading one field and writing
// another.
//
// This is the offset-aware alias analysis case, stated as a program: today a
// write to `p[i].total` invalidates every cached load at that base, so
// `p[i].a`, `p[i].b` and `p[i].c` are re-loaded on each pass even though no
// write can reach them. The instruction count here is what closing that gap
// should move.

// OUTPUT: 8
// OUTPUT: 8184
// OUTPUT: 526224
// EXIT: 0


extern int printf(char* fmt, int x);

struct Rec {
    int a;
    int b;
    int c;
    int total;
};

int main() {
    struct Rec recs[128];

    for (int i = 0; i < 128; i = i + 1) {
        recs[i].a = i * 3;
        recs[i].b = i * 5 + 1;
        recs[i].c = i % 11;
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
    printf("%d\n", recs[0].total);
    printf("%d\n", recs[127].total);
    printf("%d\n", sum);
    return 0;
}
