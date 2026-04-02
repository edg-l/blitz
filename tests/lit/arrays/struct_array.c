// RUN: %tinyc %s -o %t && %t
// EXIT: 60

struct Point {
    int x;
    int y;
};

int main() {
    struct Point pts[3];
    pts[0].x = 10;
    pts[0].y = 20;
    pts[1].x = 5;
    pts[1].y = 15;
    pts[2].x = 3;
    pts[2].y = 7;

    int sum = 0;
    int i = 0;
    while (i < 3) {
        sum = sum + pts[i].x + pts[i].y;
        i = i + 1;
    }
    return sum;
}
