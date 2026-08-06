# Saved corpus

Programs `gen_c.py` produced that blitz got wrong, kept rather than regenerated.
`bash tests/fuzz/run_corpus.sh` checks them against the same three oracles
`run_fuzz.sh` uses, in seconds.

The point is width per second. A 200-seed sweep of one shape is minutes, so it
is not run between every change; at the 30 seeds every gate *does* run, all
three shapes were green while seven programs miscompiled. **A session can work
all day, see every gate pass, and never learn that.** A program that has ever
been wrong belongs here.

Each file carries its own answer in the `// OUTPUT:` directive `gen_c.py` wrote
by interpreting the program as it built it, so a saved program does not depend
on `gen_c.py` still producing it. That is why these are files and not a list of
seeds: the generator is expected to grow new constructs, and a seed means
something different after it does.

- `fixed/` -- was wrong, now right. A failure here is a regression and fails the
  run.
- `open/` -- still wrong. Reported and counted, but does not fail the run; a
  known-open bug is not news. One that *passes* is reported loudly, and the file
  should move to `fixed/`.

| file | shape, seed | what it caught |
| --- | --- | --- |
| `fixed/pressure-seed131.c` | pressure 131 | `-O1` printed 1375 for 1411. A loop preheader carrying 16 parameters that hoisted nothing. |
| `fixed/pressure-seed165.c` | pressure 165 | `-O1` printed 3232 for 3260, same shape. |
| `fixed/pressure-seed158.c` | pressure 158 | `-O1` exited 3 where `-O0` printed -2051. |
| `fixed/args-seed52.c` | args 52 | `-O1` printed 6640 for 4025. |
| `fixed/mixed-seed109.c` | mixed 109 | `-O1` printed 675 for 1571. Slot routing named a parameter by a VReg its own block does not read. |
| `fixed/args-seed175.c` | args 175 | `-O1` printed 1298 for 544, same cause. |
| `fixed/pressure-seed128.c` | pressure 128 | `-O1` printed 1878 for 1911, same cause. |
| `fixed/mixed-seed128.c` | mixed 128 | `-O1` printed 1221 for 1178, same cause. |
| `open/mixed-seed57.c` | mixed 57 | `-O0` does not compile: register pressure overshoot where every over-budget value is one instruction's own operand. |
| `open/pressure-seed39.c` | pressure 39 | `-O1` does not compile, same shape. |
