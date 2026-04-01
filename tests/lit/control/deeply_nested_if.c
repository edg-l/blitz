// EXIT: 0
// Deeply nested if/else -- regression test for cross-block ret value fix.
int f(int a) {
    if (a > 10) {
        if (a > 20) {
            if (a > 30) {
                return 4;
            } else {
                return 3;
            }
        } else {
            return 2;
        }
    } else {
        return 1;
    }
}
int main() {
    if (f(5) != 1) { return 1; }
    if (f(15) != 2) { return 2; }
    if (f(25) != 3) { return 3; }
    if (f(35) != 4) { return 4; }
    return 0;
}
