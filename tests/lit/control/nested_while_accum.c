// EXIT: 0
// Nested while with accumulation -- regression for per-block param VReg fix.
int main() {
    int total = 0;
    int i = 0;
    while (i < 3) {
        int j = 0;
        while (j < 4) {
            total = total + (i + 1) * (j + 1);
            j = j + 1;
        }
        i = i + 1;
    }
    if (total != 60) { return 1; }
    return 0;
}
