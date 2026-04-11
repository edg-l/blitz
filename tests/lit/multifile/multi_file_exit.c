// Test multi-file with EXIT directive.
// EXTRA_FILE: helper_add.c
// EXIT: 42

extern int add(int a, int b);

int main() {
    return add(20, 22);
}
