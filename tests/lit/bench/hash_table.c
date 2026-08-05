// An open-addressed hash table of 512 slots: insert 300 keys, then look up
// every one plus 300 absent ones.
//
// Linear probing is a loop whose exit depends on memory it just read, and the
// two parallel arrays (`keys`, `vals`) are the case where knowing two bases
// cannot alias would keep a load out of the probe loop.

// OUTPUT: 300
// OUTPUT: 300
// OUTPUT: 1296029
// EXIT: 0


extern int printf(char* fmt, int x);

int probe(int* keys, int mask, int key) {
    // Unsigned so the multiply wraps rather than overflowing a signed int.
    unsigned int h = ((unsigned int)key * 2654435761) & (unsigned int)mask;
    while (keys[(int)h] != 0 && keys[(int)h] != key) {
        h = (h + 1) & (unsigned int)mask;
    }
    return (int)h;
}

int main() {
    int keys[512];
    int vals[512];
    for (int i = 0; i < 512; i = i + 1) {
        keys[i] = 0;
        vals[i] = 0;
    }

    int mask = 511;
    for (int i = 1; i <= 300; i = i + 1) {
        int key = i * 7 + 1;
        int slot = probe(keys, mask, key);
        keys[slot] = key;
        vals[slot] = i * i % 9973;
    }

    int hits = 0;
    int value_sum = 0;
    for (int i = 1; i <= 300; i = i + 1) {
        int key = i * 7 + 1;
        int slot = probe(keys, mask, key);
        if (keys[slot] == key) {
            hits = hits + 1;
            value_sum = value_sum + vals[slot];
        }
    }

    int absent = 0;
    for (int i = 1; i <= 300; i = i + 1) {
        int key = i * 7 + 4;
        int slot = probe(keys, mask, key);
        if (keys[slot] == 0) {
            absent = absent + 1;
        }
    }

    printf("%d\n", hits);
    printf("%d\n", absent);
    printf("%d\n", value_sum);
    return 0;
}
