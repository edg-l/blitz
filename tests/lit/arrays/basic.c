// RUN: %tinyc %s -o %t && %t
// EXIT: 45

int main() {
    int arr[10];
    int i = 0;
    while (i < 10) {
        arr[i] = i;
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
