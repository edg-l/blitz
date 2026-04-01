// EXIT: 0
// Short-circuit AND: if left is false, right should not be evaluated.
// Test with a pointer dereference that would segfault if evaluated.
int main() {
    int *null_ptr = (int*)0;
    // 0 && *null_ptr: if short-circuit works, null_ptr is never dereferenced.
    // Without short-circuit, this segfaults.
    if (0 && *null_ptr) { return 99; }
    return 0;
}
