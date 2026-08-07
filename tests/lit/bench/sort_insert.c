// Insertion sort over 256 pseudo-random elements.
//
// The inner loop's trip count depends on the data, so the block frequencies the
// cost model assumes are wrong for it in both directions -- and the shifting
// `arr[j + 1] = arr[j]` is a load and a store one element apart, which offset
// disjointness would have to reason about.

// OUTPUT: 1
// OUTPUT: 16
// OUTPUT: 4095
// OUTPUT: 2104214
// EXIT: 0


extern int printf(char* fmt, ...);

int main() {
    int arr[256];
    unsigned int seed = 12345;
    for (int i = 0; i < 256; i = i + 1) {
        seed = seed * 1103515245 + 12345;
        arr[i] = (int)(seed >> 16 & 4095);
    }

    for (int i = 1; i < 256; i = i + 1) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }

    int sorted = 1;
    int checksum = 0;
    for (int i = 0; i < 256; i = i + 1) {
        checksum = checksum + arr[i] * (i % 7 + 1);
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
