// `helper` has no caller in this file -- the only one is in helper_forward.c --
// so eliminating it as dead leaves an object nobody can link against.
// EXTRA_FILE: helper_forward.c
// OUTPUT: 22

extern int printf(char* fmt, ...);
extern int forward(int x);

int helper(int x) {
    return x * 3;
}

int main() {
    printf("%d\n", forward(7));
    return 0;
}
