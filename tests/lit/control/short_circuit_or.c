// EXIT: 0
// Short-circuit OR: if left is true, right should not be evaluated.
// Test with a pointer dereference that would segfault if evaluated.
int main() {
    int *null_ptr = (int*)0;
    // 1 || *null_ptr: if short-circuit works, null_ptr is never dereferenced.
    // Without short-circuit, this segfaults.
    if (1 || *null_ptr) {}
    return 0;
}
