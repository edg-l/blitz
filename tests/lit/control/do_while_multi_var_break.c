// Test: do-while with multiple variables (count and total) and break on total
// Regression: barrier sort skipped branch-only blocks, clobbering EFLAGS
// count tracks iterations, total tracks running sum; break when total > 5
// i=1 total=1 count=1, i=2 total=3 count=2, i=3 total=6>5 break (count stays 2)
// EXIT: 2

int main() {
    int total = 0;
    int count = 0;
    int i = 0;
    do {
        i = i + 1;
        total = total + i;
        if (total > 5) {
            break;
        }
        count = count + 1;
    } while (i < 10);
    return count;
}
