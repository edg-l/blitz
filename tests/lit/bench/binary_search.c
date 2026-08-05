// Binary search over a sorted 512-element array, once per key plus misses.
//
// A short loop with an unpredictable branch and a division by two the strength
// reducer should turn into a shift; the array base is invariant across every
// search, so it wants to stay in a register across the whole outer loop.

// OUTPUT: 512
// OUTPUT: 1024
// OUTPUT: 130816
// EXIT: 0


extern int printf(char* fmt, int x);

int bsearch_idx(int* arr, int n, int key) {
    int lo = 0;
    int hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == key) {
            return mid;
        }
        if (arr[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

int main() {
    int arr[512];
    for (int i = 0; i < 512; i = i + 1) {
        arr[i] = i * 3 + 1;
    }

    int hits = 0;
    int misses = 0;
    int index_sum = 0;
    for (int k = 0; k < 1536; k = k + 1) {
        int idx = bsearch_idx(arr, 512, k);
        if (idx >= 0) {
            hits = hits + 1;
            index_sum = index_sum + idx;
        } else {
            misses = misses + 1;
        }
    }
    printf("%d\n", hits);
    printf("%d\n", misses);
    printf("%d\n", index_sum);
    return 0;
}
