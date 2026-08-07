// The caller's half of a variadic call: `...` in a prototype, and the default
// argument promotions on everything past the declared parameters.
//
// Only the caller's half exists. Defining a variadic function is a parse error
// naming the callee-side gap, because the register save area SysV wants would
// need the IR to name argument registers that are not declared parameters.
//
// What this covers, in the order it is worth covering:
//
//   - int and double in one call, which is the only shape that makes AL
//     nonzero *beside* GPR arguments. `variadic_al_register.c` sets AL from a
//     call whose variadic arguments are all doubles.
//   - Zero variadic arguments, so AL=0 is deliberate rather than luck.
//   - Seven and eight integer arguments, so the tail spills to the stack at an
//     odd and an even count. That is the shape behind
//     `corpus/fixed/stack_arg_alignment.c`, reachable here from one honest
//     prototype instead of a hand-written fixed-arity stand-in.
//   - `char`, `short` and `float`, which have no variadic form of their own:
//     C17 6.5.2.2p7 promotes them to `int` and `double` at the call.
//
// `argc` keeps every value off the constant-folding path.

extern int printf(char* fmt, ...);

int main(int argc, char** argv) {
    printf("none\n");

    printf("%d\n", argc);
    printf("%d %f\n", argc, 2.5);
    printf("%f %d %f\n", 1.5, argc, 2.5);

    printf("%d %d %d %d %d %d %d\n", argc, 2, 3, 4, 5, 6, 7);
    printf("%d %d %d %d %d %d %d %d\n", argc, 2, 3, 4, 5, 6, 7, 8);
    printf("%d %d %d %d %d %d %d %d %d\n", argc, 2, 3, 4, 5, 6, 7, 8, 9);

    char c = (char)(64 + argc);
    short s = (short)(300 * argc);
    float f = 1.5f * (float)argc;
    printf("%d %d %f\n", c, s, f);

    printf("%d %f %d %f %d %f\n", argc, 1.5, argc + 1, 2.5, argc + 2, 3.5);
    return 0;
}

// OUTPUT: none
// OUTPUT: 1
// OUTPUT: 1 2.500000
// OUTPUT: 1.500000 1 2.500000
// OUTPUT: 1 2 3 4 5 6 7
// OUTPUT: 1 2 3 4 5 6 7 8
// OUTPUT: 1 2 3 4 5 6 7 8 9
// OUTPUT: 65 300 1.500000
// OUTPUT: 1 1.500000 2 2.500000 3 3.500000
// EXIT: 0
