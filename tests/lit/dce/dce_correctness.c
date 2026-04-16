// RUN: %tinyc %s -o %t && %t
// EXIT: 60

// Combined DCE correctness test with multiple branches, some dead.
// Returns 10+20+30 = 60 if the correct branches are taken.
// If dead branches (99, 88) were incorrectly taken, the result would differ.

int main() {
    int result = 0;
    int x = 1;
    if (x) {
        result = result + 10;
    } else {
        result = result + 99;
    }
    int y = 0;
    if (y) {
        result = result + 88;
    } else {
        result = result + 20;
    }
    result = result + 30;
    return result;
}
