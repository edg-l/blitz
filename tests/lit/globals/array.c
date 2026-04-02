// RUN: %tinyc %s -o %t && %t
// EXIT: 10
int arr[5];
int main() {
    int i = 0;
    while (i < 5) {
        arr[i] = i;
        i = i + 1;
    }
    int sum = 0;
    i = 0;
    while (i < 5) {
        sum = sum + arr[i];
        i = i + 1;
    }
    return sum;
}
