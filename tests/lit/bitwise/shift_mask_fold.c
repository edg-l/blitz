// RUN: %tinyc %s -o %t && %t
// EXIT: 0
// Dropping a mask the shift already made redundant must not change the value,
// including at the boundaries where the shift result is 0 or all-ones.

__attribute__((noinline))
int hi_byte(unsigned int x) { return (int)((x >> 24) & 255); }

__attribute__((noinline))
int hi_word(unsigned int x) { return (int)((x >> 16) & 65535); }

__attribute__((noinline))
int low_zeroed(unsigned int x) { return (int)((x << 8) & 4294967040); }

__attribute__((noinline))
int mid_byte(unsigned int x) { return (int)((x >> 8) & 255); }

int main() {
    unsigned int all_ones = 4294967295;
    if (hi_byte(all_ones) != 255) { return 1; }
    if (hi_byte(0) != 0) { return 2; }
    if (hi_byte(305419896) != 18) { return 3; }
    if (hi_word(all_ones) != 65535) { return 4; }
    if (hi_word(305419896) != 4660) { return 5; }
    if (low_zeroed(all_ones) != 0 - 256) { return 6; }
    if (low_zeroed(305419896) != 878082048) { return 7; }
    if (mid_byte(all_ones) != 255) { return 8; }
    if (mid_byte(305419896) != 86) { return 9; }
    return 0;
}
