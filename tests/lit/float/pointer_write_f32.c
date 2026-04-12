// Test: write through f32 pointer (store path for float32)
// Regression: OpSize::from_type panicked on F32 in effectful.rs Store path
// EXIT: 7

int main() {
    float x = 0.0f;
    float *p = &x;
    *p = 7.5f;
    int result = (int)x;
    return result;
}
