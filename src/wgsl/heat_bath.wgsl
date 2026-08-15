// The heat bath kernel fragment: one `update(v)` per variable the schedule
// hands in — price every state the variable could take through the model
// snippet's `energy_delta`, then draw one from the conditional distribution
// they define. The device counterpart of the CPU `heat_bath_step`, walking the
// same shifted-weight cumulative sum; `docs/heat-bath.md` describes the
// update.
//
// Each candidate is priced against the *current* state, which is the common
// factor the conditional is defined up to; the current state's own entry is
// written as zero rather than asked for. Shifting by the smallest delta puts
// the largest exponent at zero, so the biggest weight is exactly one and a
// strongly downhill candidate at large β cannot overflow `exp` — with a sum
// over `Q_STATES` terms there is no closed-form rearrangement to hide behind,
// so the shift is load-bearing at any state count.
//
// One keyed draw per variable per sweep, walked against the unnormalized
// cumulative sum. The last state is the structural fallback rather than a
// case: it takes every draw the earlier states do not claim, so rounding in
// the total cannot leave the walk without an answer. This is the same draw
// budget the Metropolis kernel uses at two states, so the two kernels are
// interchangeable without the host keying anything differently.

// The state count, substituted by the host: the tallies below are arrays, and
// a WGSL array length must be a const-expression, which a pipeline `override`
// is not.
const Q_STATES: u32 = $Q$u;

fn update(v: u32) {
    let current = vars[v];

    var deltas: array<f32, Q_STATES>;
    var lowest = 0.0;
    for (var s = 0u; s < Q_STATES; s = s + 1u) {
        var dE = 0.0;
        if (s != current) {
            dE = energy_delta(v, current, s);
        }
        deltas[s] = dE;
        lowest = min(lowest, dE);
    }

    var weights: array<f32, Q_STATES>;
    var total = 0.0;
    for (var s = 0u; s < Q_STATES; s = s + 1u) {
        weights[s] = exp(-params.beta * (deltas[s] - lowest));
        total = total + weights[s];
    }

    // The name avoids `target`, which WGSL reserves.
    let threshold = keyed_uniform(params.seed, v, pc.sweep) * total;
    var chosen = Q_STATES - 1u;
    var cumulative = 0.0;
    for (var s = 0u; s + 1u < Q_STATES; s = s + 1u) {
        cumulative = cumulative + weights[s];
        if (threshold < cumulative) {
            chosen = s;
            break;
        }
    }

    vars[v] = chosen;
}
