// RUN: %tinyc %s -o %t && %t
// EXIT: 225

int main() {
    int arr[50];
    int i = 0;
    while (i < 50) {
        arr[i] = i;
        i = i + 1;
    }
    // Sum all: 0+1+...+49 = 1225, take mod 256 for exit code: 1225 % 256 = 201
    // Actually just sum first 50 to get a known value
    int sum = 0;
    i = 0;
    while (i < 50) {
        sum = sum + arr[i];
        i = i + 1;
    }
    // 0+1+2+...+49 = 1225, mod 256 = 201
    // Return lower byte via modular reduction
    return sum - 1000;
}
