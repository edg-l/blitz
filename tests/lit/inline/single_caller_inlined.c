// RUN: %tinyc %s --emit-ir 2>&1
// A helper called exactly once should always be inlined regardless of size.
// CHECK-LABEL: function main
// CHECK-NOT: call helper
// CHECK: ret

// RUN: %tinyc %s -o %t && %t
// EXIT: 34

int helper(int x) {
    int a = x + 1;
    int b = a * 2;
    int c = b + a;
    int d = c * c;
    int e = d / (a + 1);
    int f = e - b;
    return f;
}

int main() {
    return helper(5);
}
