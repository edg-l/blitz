// EXIT: 0
// OUTPUT: 0
// OUTPUT: 1
// OUTPUT: 2
// OUTPUT: 3
// OUTPUT: 4

extern int printf(char* fmt, int x);
int main() {
    for (int i = 0; i < 5; i = i + 1) {
        printf("%d\n", i);
    }
    return 0;
}
