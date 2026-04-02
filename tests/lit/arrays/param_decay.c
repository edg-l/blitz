// RUN: %tinyc %s -o %t && %t
// EXIT: 15

__attribute__((noinline))
int fill_and_sum(int *arr, int n) {
    int i = 0;
    while (i < n) {
        arr[i] = i + 1;
        i = i + 1;
    }
    int total = 0;
    i = 0;
    while (i < n) {
        total = total + arr[i];
        i = i + 1;
    }
    return total;
}

int main() {
    int arr[5];
    return fill_and_sum(arr, 5);
}
