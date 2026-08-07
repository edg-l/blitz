// An insertion sort, then a binary search over what it produced.
//
// Two shapes no other kernel here has. The sort's inner loop is a shift chain
// whose trip count depends on the data, so its bound cannot be hoisted and the
// load of `a[j]` is re-read every iteration against a store to `a[j + 1]` --
// the one place alias analysis has to decide whether the store kills the load.
// The search is a bisection: the loop body is three comparisons and a shift
// with no memory traffic between them, so it is judged on whether the index
// arithmetic collapses into the addressing mode and whether the two-way update
// of `lo`/`hi` becomes a conditional move or a branch.
//
// Both phases run on data seeded from `argc`, so the sorted order is not a
// constant any reference compiler can fold.

// OUTPUT: 48368
// OUTPUT: 998
// OUTPUT: 2015
// OUTPUT: 359
// EXIT: 0

extern int printf(char* fmt, int x);

int main(int argc, char** argv) {
    int a[96];
    unsigned int s = (unsigned int)argc * 2654435761 + 12345;

    for (int i = 0; i < 96; i = i + 1) {
        s = s * 1103515245 + 12345;
        a[i] = (int)((s >> 16) & 1023);
    }

    for (int i = 1; i < 96; i = i + 1) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j = j - 1;
        }
        a[j + 1] = key;
    }

    int sum = 0;
    int top = a[95];
    for (int i = 0; i < 96; i = i + 1) {
        sum = sum + a[i];
    }

    int hits = 0;
    int steps = 0;
    for (int t = 0; t < 64; t = t + 1) {
        int target = a[(t * 7) & 63];
        int lo = 0;
        int hi = 95;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            steps = steps + 1;
            if (a[mid] == target) {
                hits = hits + mid;
                break;
            }
            if (a[mid] < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }

    printf("%d\n", sum);
    printf("%d\n", top);
    printf("%d\n", hits);
    printf("%d\n", steps);
    return 0;
}
