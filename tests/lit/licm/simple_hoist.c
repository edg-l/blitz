// RUN: %tinyc %s --enable-licm --emit-ir 2>&1
// CHECK-LABEL: function main
// Preheader block should exist (block with params that jumps to the loop header).
// The iconst(10) and iconst(1) used in the loop should appear in the preheader,
// not only in the loop body.
// CHECK: block{{[0-9]+}}
// CHECK: jump block
// CHECK: branch
// CHECK: ret

// Verify correctness: (3+4) * 10 = 70
// RUN: %tinyc %s --enable-licm -o %t && %t
// EXIT: 70

int main() {
    int sum = 0;
    int i = 0;
    int x = 3;
    int y = 4;
    while (i < 10) {
        int z = x + y;
        sum = sum + z;
        i = i + 1;
    }
    return sum;
}
