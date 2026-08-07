// A `u` suffix makes a literal unsigned, and everything downstream has to agree.
//
// Without the suffix there was no way to write an unsigned value in tinyc at
// all, so the unsigned condition codes, the logical shift and the unsigned
// divide were unreachable from source. Three bugs were hiding behind that:
//
//   - An `Iconst` of I32 carries the 32-bit pattern, so `4294967295u` rides as
//     `-1`. Emitting it as the i64 `4294967295` is out of range for the type.
//   - `Builder::icmp` moves a constant to the right and flips the condition
//     with it, so `1u < u` is emitted as `u > 1`. A `select` built with the
//     condition that went *in* tests `Ult` against the flags of `cmp u, 1` and
//     answers `u < 1`. `icmp_canonical` returns the condition that came out.
//   - Constant folding of a logical shift ran on the sign-extended 64-bit
//     pattern, so `0xffffffffu >> 28` brought down sign bits and folded to
//     `0xffffffff` instead of `15`.
//
// `argc` keeps the second group off the folding path, so both are covered.

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    unsigned int folded = 4294967295u;
    printf("%d\n", (int)(folded >> 28));
    printf("%d\n", (int)(folded / 3u));

    unsigned int big = 4294967288u * (unsigned int)argc;
    unsigned int one = 1u;
    printf("%d\n", one < big);
    printf("%d\n", big > one);
    printf("%d\n", big < one);
    printf("%d\n", (int)(big >> 28));

    int neg = 0 - 1;
    printf("%d\n", neg < (int)one);
    return 0;
}

// OUTPUT: 15
// OUTPUT: 1431655765
// OUTPUT: 1
// OUTPUT: 1
// OUTPUT: 0
// OUTPUT: 15
// OUTPUT: 1
// EXIT: 0
