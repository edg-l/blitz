// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// EXIT: 30
//
// A pair-producing x86 op keeps its result in the pair VReg and `Proj0` reads
// it out, so lowering emits `mov proj, pair` unless the two get one register.
// They can: `interference::result_shares_operand` keeps the edge between them
// out of the graph, so the accumulation happens in place.

// CHECK-LABEL: # main
// The accumulation, with no copy setting up either operand.
// CHECK: add    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK-NOT: mov    {{[a-z0-9]+}},{{[a-z0-9]+}}
// CHECK: inc    {{[a-z0-9]+}}

int main() {
    int sum = 0;
    for (int i = 0; i < 5; i = i + 1) {
        sum = sum + i * 3;
    }
    return sum;
}
