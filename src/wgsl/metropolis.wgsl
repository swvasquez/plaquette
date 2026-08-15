// The Metropolis kernel fragment: one `update(v)` per variable the schedule
// hands in — propose an alternative state, price it through the model
// snippet's `energy_delta`, and accept with `min(1, e^{-β ΔE})`. The device
// counterpart of the CPU `step`, and like it blind to the model and to the
// grade: `v` is a bare index into `vars`.
//
// The proposal is drawn uniformly from the `Q_STATES - 1` states that are not
// the current one, mapped onto them by skipping past the current index — a
// bijection, so no state is offered twice or not at all, which is the symmetry
// the acceptance rule rests on when it drops the Hastings ratio (see
// `docs/metropolis.md`). At two states the proposal is the deterministic flip
// and consumes no draw; the branch is on a substituted constant, so the
// compiler resolves it before a thread ever runs.
//
// Randomness is counter-based: the draws are pure functions of
// (seed, variable, sweep), so the result is independent of thread order. The
// proposal draw is keyed off a different stream from the accept draw — a
// variable that picked its candidate and its acceptance from the same number
// would accept exactly the states the mixing happens to send low.

// The state count, substituted by the host — always known there, since `Q` is
// a compile-time parameter of the chain. A constant rather than a uniform
// because the two-state branch below should cost nothing, and because the heat
// bath fragment's arrays need a const length for the same token.
const Q_STATES: u32 = $Q$u;

// Keys the proposal draw off a different stream from the accept draw. Any
// constant unrelated to the ones inside `keyed_uniform` does the job, since
// the seed is mixed before it is used.
const PROPOSAL_KEY: u32 = 0x85ebca6bu;

fn update(v: u32) {
    let current = vars[v];

    var proposed: u32;
    if (Q_STATES == 2u) {
        // The flip, with no draw: consuming randomness for a determined
        // outcome would shift every existing two-state run onto different
        // draws while sampling exactly the same distribution.
        proposed = 1u - current;
    } else {
        // The clamp is not paranoia: `keyed_uniform` returns the largest float
        // below one, and scaling that by `Q_STATES - 1` can round *up* to
        // exactly `Q_STATES - 1` once it exceeds two, which would put the
        // proposal one past the last state.
        let alternatives = Q_STATES - 1u;
        var draw = u32(keyed_uniform(params.seed ^ PROPOSAL_KEY, v, pc.sweep) * f32(alternatives));
        if (draw >= alternatives) {
            draw = alternatives - 1u;
        }
        proposed = draw;
        if (proposed >= current) {
            proposed = proposed + 1u;
        }
    }

    let dE = energy_delta(v, current, proposed);

    let u = keyed_uniform(params.seed, v, pc.sweep);
    if (dE <= 0.0 || u < exp(-params.beta * dE)) {
        vars[v] = proposed;
    }
}
