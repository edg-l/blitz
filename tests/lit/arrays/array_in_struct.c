// RUN: %tinyc %s -o %t && %t
// EXIT: 30

struct Data {
    int values[4];
    int count;
};

int main() {
    struct Data d;
    d.count = 4;
    d.values[0] = 3;
    d.values[1] = 7;
    d.values[2] = 11;
    d.values[3] = 9;

    int sum = 0;
    int i = 0;
    while (i < d.count) {
        sum = sum + d.values[i];
        i = i + 1;
    }
    return sum;
}
