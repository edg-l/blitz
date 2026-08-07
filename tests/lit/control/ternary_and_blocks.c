// A conditional expression is an argument, and a bare block is a scope.
//
// The ternary parsed everywhere except where it is most often written. Call
// arguments parse at binding power 1 so the comma stays a separator, and the
// ternary broke out at that same level -- but C17 6.5.2.2p1 makes an argument
// an assignment-expression, which includes a conditional-expression. So
// `f(a ? b : c)` was a syntax error while `x = a ? b : c` was not.
//
// A bare `{ ... }` was not a statement at all, which is also the only way to
// write a scope that is not a loop or an `if`.

extern int printf(char* fmt, ...);

__attribute__((noinline))
int pick(int x) {
    return x;
}

int main(int argc, char** argv) {
    int a = argc;

    printf("%d\n", a > 0 ? 10 : 20);
    printf("%d\n", a > 5 ? 10 : 20);
    printf("%d\n", pick(a == 1 ? 7 : 8));
    printf("%d\n", a > 0 ? (a > 5 ? 1 : 2) : 3);

    int v = 7;
    {
        int v = 50;
        printf("%d\n", v);
        {
            int v = 900;
            printf("%d\n", v);
        }
        printf("%d\n", v);
    }
    printf("%d\n", v);
    return 0;
}

// OUTPUT: 10
// OUTPUT: 20
// OUTPUT: 7
// OUTPUT: 2
// OUTPUT: 50
// OUTPUT: 900
// OUTPUT: 50
// OUTPUT: 7
// EXIT: 0
