// Test that array bracket syntax in parameters decays to pointer.
// BUG: constant-index array stores + call in same block segfaults (StackAddr scheduling bug)
// RUN: %tinyc %s -o %t && %t
// EXIT: 10

__attribute__((noinline))
int sum(int arr[5], int n) {
    int total = 0;
    int i = 0;
    while (i < n) {
        total = total + arr[i];
        i = i + 1;
    }
    return total;
}

int main() {
    int data[4];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 4;
    return sum(data, 4);
}
