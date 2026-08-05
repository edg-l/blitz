// Recursive quicksort over 256 pseudo-random elements.
//
// Recursion with an array parameter: the pointer is live across both recursive
// calls, so it must hold a callee-saved register or a slot, and the partition
// loop is the classic two-pointer walk over one buffer.

// OUTPUT: 1
// OUTPUT: 32
// OUTPUT: 9999
// OUTPUT: 1269635
// EXIT: 0


extern int printf(char* fmt, int x);

int partition(int* arr, int lo, int hi) {
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j = j + 1) {
        if (arr[j] <= pivot) {
            i = i + 1;
            int t = arr[i];
            arr[i] = arr[j];
            arr[j] = t;
        }
    }
    int t = arr[i + 1];
    arr[i + 1] = arr[hi];
    arr[hi] = t;
    return i + 1;
}

void quicksort(int* arr, int lo, int hi) {
    if (lo < hi) {
        int p = partition(arr, lo, hi);
        quicksort(arr, lo, p - 1);
        quicksort(arr, p + 1, hi);
    }
}

int main() {
    int arr[256];
    unsigned int seed = 987654321;
    for (int i = 0; i < 256; i = i + 1) {
        seed = seed ^ (seed << 13);
        seed = seed ^ (seed >> 17);
        seed = seed ^ (seed << 5);
        arr[i] = (int)(seed % 10007);
    }

    quicksort(arr, 0, 255);

    int sorted = 1;
    int checksum = 0;
    for (int i = 0; i < 256; i = i + 1) {
        checksum = checksum + arr[i];
        if (i > 0 && arr[i - 1] > arr[i]) {
            sorted = 0;
        }
    }
    printf("%d\n", sorted);
    printf("%d\n", arr[0]);
    printf("%d\n", arr[255]);
    printf("%d\n", checksum);
    return 0;
}
