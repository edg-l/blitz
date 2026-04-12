// Test: nested do-while with break in inner loop
// Regression: barrier sort skipped branch-only blocks, clobbering EFLAGS
// Outer: i=0..2 (3 iterations). Inner: j counts up, breaks at j==2.
// Each outer iteration adds 2 to sum (j=1 then j=2 breaks before adding 2).
// Wait: inner adds j before break check.
// j=1: sum+=1, j=2: break. So inner adds 1 per outer iteration.
// 3 outer iterations => sum = 3
// EXIT: 3

int main() {
    int sum = 0;
    int i = 0;
    do {
        int j = 0;
        do {
            j = j + 1;
            if (j == 2) {
                break;
            }
            sum = sum + j;
        } while (j < 10);
        i = i + 1;
    } while (i < 3);
    return sum;
}
