// EXIT: 0
// Test ++/-- in complex contexts

__attribute__((noinline)) int id(int x) { return x; }

int main() {
    // Post-increment in function call argument
    int a = 5;
    int r = id(a++);
    if (r != 5) { return 1; }  // old value passed
    if (a != 6) { return 2; }  // a was incremented

    // Pre-increment in function call argument
    int b = 5;
    int s = id(++b);
    if (s != 6) { return 3; }  // new value passed
    if (b != 6) { return 4; }

    // Increment via pointer dereference
    int val = 10;
    int *p = &val;
    ++(*p);
    if (val != 11) { return 5; }
    (*p)++;
    if (val != 12) { return 6; }

    // Decrement via pointer dereference
    --(*p);
    if (val != 11) { return 7; }
    (*p)--;
    if (val != 10) { return 8; }

    // Multiple increments in one expression
    int c = 1;
    int d = ++c + ++c;  // c becomes 2, then 3; d = 2 + 3 = 5
    if (c != 3) { return 9; }
    if (d != 5) { return 10; }

    // Increment in for-loop with compound body
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += i;
    }
    // sum = 0+1+2+...+9 = 45
    if (sum != 45) { return 11; }

    // Countdown with post-decrement
    int arr[5];
    int idx = 4;
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    int total = 0;
    while (idx >= 0) {
        total += arr[idx];
        idx--;
    }
    if (total != 150) { return 12; }

    return 0;
}
