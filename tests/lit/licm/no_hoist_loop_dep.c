// RUN: %tinyc %s --enable-licm -o %t && %t
// EXIT: 55

// Loop-carried dependency: sum depends on previous iteration's sum.
// The accumulation cannot be hoisted.

int main() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        sum = sum + i + 1;
        i = i + 1;
    }
    return sum;
}
