// RUN: %tinyc %s --enable-licm -o %t && %t
// EXIT: 55

// Effectful ops (loads) should NOT be hoisted out of the loop.
// Each iteration loads from the pointer, which could change.

int main() {
    int arr[10];
    int i = 0;
    while (i < 10) {
        arr[i] = i + 1;
        i = i + 1;
    }
    int sum = 0;
    i = 0;
    while (i < 10) {
        sum = sum + arr[i];
        i = i + 1;
    }
    return sum;
}
