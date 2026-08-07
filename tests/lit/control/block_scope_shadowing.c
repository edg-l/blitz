// A declaration inside a block shadows an outer one for that block, and only
// for that block.
//
// tinyc's name maps are flat and keyed by name, so a declaration in a body used
// to displace the outer binding for the rest of the function. The loop below is
// what makes that visible rather than merely wrong: `int r = 100` re-entered
// the loop counter's own binding every iteration, the counter never advanced,
// and the program hung. gcc prints 300.
//
// The second loop checks the other half of the rule -- that the outer binding
// comes back after the block, rather than staying displaced.

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    int total = 0;
    for (int r = 0; r < 3 * argc; r = r + 1) {
        int r = 100;
        total = total + r;
    }
    printf("%d\n", total);

    int v = 7;
    if (argc > 0) {
        int v = 50;
        total = total + v;
    }
    printf("%d\n", v);

    int sum = 0;
    for (int i = 0; i < 4; i = i + 1) {
        int i = 1000;
        sum = sum + i;
    }
    printf("%d\n", sum);
    return 0;
}

// OUTPUT: 300
// OUTPUT: 7
// OUTPUT: 4000
// EXIT: 0
