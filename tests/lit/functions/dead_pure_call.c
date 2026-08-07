// A call to a pure function whose result nobody reads is not emitted.
//
// The result is the only thing calling a pure function can produce, so a call
// whose results are unread computes nothing observable. This is the call-shaped
// case of dead-load elimination, and it needs its own pass because a call the
// inliner declined is otherwise the one dead computation nothing removes: the
// e-graph never sees it, since an effectful op is what it is by construction.
//
// Purity is a module-level fact and it is the greatest fixpoint, so
// `mutually_pure` below stays pure through its own recursion. `stores` is impure
// because it stores, and `shouts` because it calls something this module does not
// define and therefore cannot see into.
//
// RUN: %tinyc %s -o %t --emit-asm | %blitztest %s
// CHECK-LABEL: # main
// Exactly four calls survive -- `stores`, `shouts`, the `pure_leaf` whose result
// is read, and the `printf` that reads it. `CHECK-COUNT-N` is exact, so this
// asserts the two dead calls are gone as well as that these four are not.
// CHECK-COUNT-4: call

extern int printf(char* fmt, ...);

__attribute__((noinline))
int pure_leaf(int a, int b) {
    return (a * 3 + b) & 1023;
}

__attribute__((noinline))
int mutually_pure(int a) {
    if (a > 100) {
        return mutually_pure(a - 7) + pure_leaf(a, 1);
    }
    return pure_leaf(a, 2);
}

__attribute__((noinline))
int stores(int a) {
    int buf[4];
    buf[a & 3] = a;
    return buf[0];
}

__attribute__((noinline))
int shouts(int a) {
    printf("%d\n", a);
    return a;
}

int main(int argc, char** argv) {
    int dead0 = pure_leaf(argc, 9);
    int dead1 = mutually_pure(argc + 200);
    int alive0 = stores(argc);
    int alive1 = shouts(argc + 40);
    int kept = pure_leaf(argc, 5);
    printf("%d\n", kept + (alive0 - alive0) + (alive1 - alive1));
    return 0;
}

// OUTPUT: 41
// OUTPUT: 8
// EXIT: 0
