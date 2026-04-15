# E-Graph Optimization Roadmap

Status of rewrite rules and e-class analyses in Blitz's e-graph.

## Implemented

### Algebraic Rules (`algebraic.rs`)
- Identity: `Add(a,0)=a`, `Mul(a,1)=a`, `Or(a,0)=a`, `And(a,-1)=a`
- Annihilation: `Mul(a,0)=0`, `And(a,0)=0`
- Idempotence: `And(a,a)=a`, `Or(a,a)=a`
- Inverse: `Sub(a,a)=0`, `Xor(a,a)=0`
- Double negation: `Sub(0, Sub(0, a))=a`
- Constant folding: all binary ops with two constant children
- Commutativity: `Add`, `Mul`, `And`, `Or`, `Xor` (canonical ordering guard)
- Reassociation: `(a+b)+c = a+(b+c)` when b,c constant and a non-constant
- Shift combining: `Shl(Shl(a,n),m) = Shl(a,n+m)` (same for Shr, Sar)
- Absorption: `And(a, Or(a,b))=a`, `Or(a, And(a,b))=a`

### Strength Reduction (`strength.rs`)
- `Mul(a, 2^n)` -> `Shl(a, n)`
- `Mul(a, 3/5/9)` -> `Add(a, Shl(a, 1/2/3))`
- `UDiv(a, 2^n)` -> `Shr(a, n)`
- `URem(a, 2^n)` -> `And(a, 2^n-1)`
- `SDiv(a, 2^n)` -> arithmetic shift pattern (I64 only)

### Instruction Selection (`isel.rs`)
- ALU: Add/Sub/And/Or/Xor -> X86 variants
- Shifts: Shl/Shr/Sar -> X86Shl/X86Shr/X86Sar + immediate variants
- Compares: Icmp -> Proj1(X86Sub), Fcmp -> X86Ucomisd/X86Ucomiss
- Division: SDiv/SRem -> Proj0/Proj1(X86Idiv), UDiv/URem -> Proj0/Proj1(X86Div)
- FP: Fadd/Fsub/Fmul/Fdiv -> X86Addsd/ss etc.
- Conversions: Sext/Zext/Trunc/Bitcast/F2I/I2F/FpromoteF/Fdemote
- Select -> X86Cmov

### Addressing Modes (`addr_mode.rs`)
- `Add(base, Iconst(d))` -> `Addr{scale:1, disp:d}`
- `Add(base, Shl(idx, n))` -> `Addr{scale:2^n, disp:0}`
- `Add(base, Mul(idx, s))` -> `Addr{scale:s, disp:0}` for s in {2,4,8}
- `Add(base, idx)` -> `Addr{scale:1, disp:0}`
- LEA2: `Add(a, b)` -> `X86Lea2(a, b)`
- LEA3: `Add(a, Shl(b, n))` -> `X86Lea3{scale:2^n}(a, b)`, `Mul(a, 3/5/9)` -> `X86Lea3`
- LEA4: `Add(Add(a, Shl(b, n)), Iconst(d))` -> `X86Lea4{scale, disp}`
- LEA4: `Add(a, Iconst(d))` -> `X86Lea4{scale:1, disp:d}`

### Comparison Folding (`algebraic.rs`)
- Same-operand: `Select(Icmp(Eq,a,a), t, f)=t`, `Select(Icmp(Ne,a,a), t, f)=f`, etc.
- Constant-constant: `Select(Icmp(cc, c1, c2), t, f)` evaluated at compile time

### Distributivity/Factoring (`distributive.rs`)
- `Add(Mul(a,b), Mul(a,c)) = Mul(a, Add(b,c))` with blowup guard
- `Sub(Mul(a,b), Mul(a,c)) = Mul(a, Sub(b,c))` with blowup guard

### E-Class Analyses
- Constant analysis: `Option<(i64, Type)>` per class, maintained in add/merge
- Known-bits analysis: `KnownBits { known_zeros: u64, known_ones: u64 }` per class, maintained in add/merge, propagated through And/Or/Xor/Shl/Shr/Sar/Zext/Sext/Trunc

### Known-Bits Exploitation (`known_bits.rs`)
- Redundant And removal: `And(x, mask)=x` when mask doesn't clear any possibly-set bits
- Known-constant promotion: when all bits are determined, add Iconst and merge

### Previously Added Algebraic Rules
- Division/remainder identities: `SDiv(a,1)=a`, `SRem(a,1)=0`, `URem(a,1)=0`, `UDiv(a,1)=a`, `SDiv(a,-1)=Sub(0,a)`
- Select simplifications: `Select(c,a,a)=a`
- Extension folding: `Zext(Zext(a))=Zext(a)`, `Sext(Sext(a))=Sext(a)`, `Trunc(Trunc(a))=Trunc(a)`
- Bitwise complement: `Or(a, Xor(a,-1))=-1`, `And(a, Xor(a,-1))=0`
- De Morgan's laws: `Xor(And(a,b),-1)=Or(Xor(a,-1),Xor(b,-1))`, `Xor(Or(a,b),-1)=And(Xor(a,-1),Xor(b,-1))`
- Negation distribution: `Sub(0, Add(a,b))=Add(Sub(0,a), Sub(0,b))`
- Or annihilation: `Or(a,-1)=-1`

## TODO

### Additional Rules
- [ ] Shift + mask optimization: `And(Shr(a,n), mask)` when shift zeroes masked bits

### Type-Width Analysis
- Domain: `(min_bits: u8, signed: bool)` per class
- Enables: narrowing operations, avoiding unnecessary sign/zero extensions

## Algorithm Compliance (egg paper)

| Component | Status | Notes |
|-----------|--------|-------|
| add() with hash-consing | Done | Canonicalizes children, checks memo |
| merge() with analysis join | Done | Joins constant_value and known_bits |
| rebuild() congruence closure | Done | Full memo drain + re-canonicalize |
| Unified saturation loop | Done | All rule categories per iteration |
| E-class analysis: make | Done | constant_value and known_bits in add() |
| E-class analysis: merge | Done | constant_value and known_bits join in merge() |
| E-class analysis: modify | Done | Known-bits propagation + constant promotion |
| Blowup protection | Done | max_classes limit |
| Cost-based extraction | Done | Bottom-up DP |
