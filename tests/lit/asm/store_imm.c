// RUN: %tinyc -O1 --emit-asm %s | %blitztest %s
// A constant stored to memory is the store's immediate, not a register the
// constant is first materialized into.

struct P { int x; int y; long tag; };

int store_consts(struct P *p, int n) {
  p->x = 3;
  p->y = 7;
  p->tag = 11;
  return p->x + n;
}

int main() {
  struct P q;
  return store_consts(&q, 39);
}

// CHECK-LABEL: # main
// CHECK: mov    DWORD PTR [{{[a-z0-9]+}}],0x3
// CHECK: mov    DWORD PTR [{{[a-z0-9]+}}+0x4],0x7
// CHECK-NOT: mov    {{e[a-z0-9]+}},0x7

// EXIT: 42
