// A four-state machine driven by a runtime byte stream, plus the byte copy that
// records what it accepted.
//
// The state variable is loop-carried through an if/else chain, so every
// iteration is a branch on a value the previous iteration computed: no
// prediction the compiler can make statically, and a live range that crosses
// every arm. The copy loop at the end is the memcpy shape, one byte at a time.

// OUTPUT: 52726
// OUTPUT: 639426
// OUTPUT: 639426
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    unsigned char stream[512];
    unsigned char kept[512];
    int chk0 = 0;
    int chk1 = 0;
    int chk2 = 0;
    int reps = 643 * argc;
    for (int r = 0; r < reps; r = r + 1) {
        int seed = ((argc + r) * 23) & 127;

        for (int i = 0; i < 512; i = i + 1) {
            stream[i] = (unsigned char)((i * 41 + seed) & 255);
            kept[i] = 0;
        }

        int state = 0;
        int accepted = 0;
        for (int i = 0; i < 512; i = i + 1) {
            int c = (int)stream[i];
            if (state == 0) {
                if (c & 1) {
                    state = 1;
                } else {
                    state = 2;
                }
            } else {
                if (state == 1) {
                    if (c > 128) {
                        state = 3;
                    } else {
                        state = 0;
                    }
                } else {
                    if (state == 2) {
                        if ((c & 7) == 0) {
                            state = 3;
                        } else {
                            state = 1;
                        }
                    } else {
                        kept[accepted] = stream[i];
                        accepted = accepted + 1;
                        state = 0;
                    }
                }
            }
        }

        int copied = 0;
        unsigned char out[512];
        for (int i = 0; i < accepted; i = i + 1) {
            out[i] = kept[i];
            copied = copied + 1;
        }

        int sum = 0;
        for (int i = 0; i < copied; i = i + 1) {
            sum = sum + (int)out[i];
        }

        int check = 0;
        for (int i = 0; i < accepted; i = i + 1) {
            check = check + (int)kept[i];
        }

        chk0 = (chk0 + (accepted)) & 1048575;
        chk1 = (chk1 + (sum)) & 1048575;
        chk2 = (chk2 + (check)) & 1048575;
    }
    printf("%d\n", chk0);
    printf("%d\n", chk1);
    printf("%d\n", chk2);
    return 0;
}
