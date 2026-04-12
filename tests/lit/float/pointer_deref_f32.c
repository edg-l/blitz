// Test: f32 pointer dereference
// Known bug: OpSize::from_type panics on F32 in effectful.rs
// EXIT: 0

int main() {
    float x = 2.5f;
    float *p = &x;
    float y = *p;
    if (y > 2.0f) {
        if (y < 3.0f) {
            return 0;
        }
    }
    return 1;
}
