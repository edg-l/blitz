// Test: do-while with both break and continue in the same loop
// Regression: barrier sort skipped branch-only blocks, clobbering EFLAGS
// i iterates 1..10; continue at i==3 (skip adding 3), break at i==5
// sum = 1 + 2 + 4 = 7
// EXIT: 7

int main() {
    int sum = 0;
    int i = 0;
    do {
        i = i + 1;
        if (i == 3) {
            continue;
        }
        if (i == 5) {
            break;
        }
        sum = sum + i;
    } while (i < 10);
    return sum;
}
