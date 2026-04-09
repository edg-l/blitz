// EXIT: 0
// Test IEEE 754 NaN comparison semantics.
// Use noinline function to prevent e-graph from merging NaN's fdiv class.

__attribute__((noinline)) double make_nan() {
    return 0.0 / 0.0;
}

int main() {
    double nan = make_nan();
    double one = 1.0;

    // NaN == NaN should be false (0)
    int eq = (nan == nan);
    if (eq != 0) { return 1; }

    // NaN != NaN should be true (1)
    int ne = (nan != nan);
    if (ne != 1) { return 2; }

    // NaN < 1.0 should be false
    int lt = (nan < one);
    if (lt != 0) { return 3; }

    // NaN > 1.0 should be false
    int gt = (nan > one);
    if (gt != 0) { return 4; }

    // NaN <= 1.0 should be false
    int le = (nan <= one);
    if (le != 0) { return 5; }

    // NaN >= 1.0 should be false
    int ge = (nan >= one);
    if (ge != 0) { return 6; }

    // 1.0 < NaN should be false
    int lt2 = (one < nan);
    if (lt2 != 0) { return 7; }

    // 1.0 > NaN should be false
    int gt2 = (one > nan);
    if (gt2 != 0) { return 8; }

    // 1.0 == NaN should be false
    int eq2 = (one == nan);
    if (eq2 != 0) { return 9; }

    // 1.0 != NaN should be true
    int ne2 = (one != nan);
    if (ne2 != 1) { return 10; }

    // Normal comparisons should still work
    double a = 3.14;
    double b = 2.71;
    if (!(a > b)) { return 11; }
    if (!(a >= b)) { return 12; }
    if (!(b < a)) { return 13; }
    if (!(b <= a)) { return 14; }
    if (!(a == a)) { return 15; }
    if (a != a) { return 16; }

    return 0;
}
