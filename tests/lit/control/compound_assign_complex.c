// EXIT: 0
// Test compound assignment in complex contexts

struct Point { int x; int y; };

int main() {
    // Compound assign with array index
    int arr[5];
    arr[0] = 10;
    arr[1] = 20;
    arr[2] = 30;
    arr[3] = 40;
    arr[4] = 50;
    arr[2] += 5;
    if (arr[2] != 35) { return 1; }
    arr[4] -= 10;
    if (arr[4] != 40) { return 2; }

    // Compound assign with struct field
    struct Point p;
    p.x = 10;
    p.y = 20;
    p.x += 5;
    p.y *= 3;
    if (p.x != 15) { return 3; }
    if (p.y != 60) { return 4; }

    // Chained compound assignment
    int a = 100;
    a += 10;
    a -= 5;
    a *= 2;
    a /= 3;
    // (100+10-5)*2/3 = 210/3 = 70
    if (a != 70) { return 5; }

    // Compound assign in for-loop
    int total = 1;
    for (int i = 0; i < 5; i++) {
        total *= 2;
    }
    // 1*2^5 = 32
    if (total != 32) { return 6; }

    return 0;
}
