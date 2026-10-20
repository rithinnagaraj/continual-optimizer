# POST-P1-A2 — CAP state-conditioned knob-liveness retry

**Timing:** written after the isolated-v2 knob-liveness preflight reported
`KNOB-INERT cap_release k` at ten cold-start steps, before any retention-grid
cell ran.

## Mis-specification

The generic ten-step liveness check starts CAP with `imp = 0`.  At that state
the box radius `k / (sqrt(imp) + eps)` is enormous for both registered `k`
values, so neither box binds and changing `k` cannot yet change parameters.
The zero distance therefore tests whether protection has naturally activated
within ten steps, not whether the constructor consumes `k` or whether `k`
affects active CAP protection.

The same issue can affect the state-dependent knobs `T_a` and `T_m`.

## Correction

The preflight retains the original cold-start ten-step comparison.  If that
comparison is exactly zero for `k`, `T_a`, or `T_m` on a CAP arm, it retries
the same two configurations from identical, deterministically primed CAP
state:

1. perform one identical optimizer step with only the hard box clamp disabled;
2. set each protected parameter's anchor to its current value and importance
   to 1.0 in both arms;
3. restore the ordinary box clamp;
4. run ten identical measured optimizer steps;
5. if the distance remains exactly zero, retry with 200 measured steps so
   slow time-constant effects can rise above float32 resolution; and
6. require the final parameter-vector L2 distance to be greater than zero.

The two arms still differ only in the knob under test.  Priming changes only
the diagnostic's starting state; it does not change any grid configuration,
training cell, validation protocol, stream protocol, threshold, or selection
rule.  The liveness record reports `state_primed=true` and names this amendment
whenever the retry is used.
