// Checkerboard heat bath sweep for the Ising model (Q = 2), in any dimension.
//
// The sibling of this model's `checkerboard.wgsl`, and deliberately identical to
// it down to the point where the energy change is in hand: same bindings, same
// bounds check, same color guard, same neighbor scan. The coloring belongs to
// the schedule rather than to the kernel, so a heat bath sweep needs exactly the
// independence a Metropolis sweep needs and gets it the same way — one dispatch
// per color, no two sites of a color neighbors, every read landing on the other
// color. See `checkerboard.wgsl` for that argument in full; it is not repeated
// here.
//
// What differs is the last three lines. Metropolis proposes the flip and may
// reject it; the heat bath asks instead for the probability that the site ends
// up flipped, given the neighbors it is conditioning on, and writes that outcome
// whatever it is. At two states the conditional needs no array and no
// normalization pass — the two weights are `1` and `exp(-b dE)`, so the flipped
// state's share collapses to a single expression. `docs/heat-bath.md` describes
// the update.
//
// Randomness is counter-based on `(seed, site, sweep)` as everywhere else, and
// the heat bath uses exactly one draw per site per sweep — the same budget the
// Metropolis kernel uses, so the two are interchangeable without the host
// keying anything differently.

override D: u32;

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
    h: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    color: u32,   // which color this dispatch updates (0 or 1)
};

@group(0) @binding(0) var<storage, read_write> spins: array<u32>;      // 0 or 1 per site
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;        // 2D per site: +0,-0,+1,-1,...
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> site_color: array<u32>;       // coordinate-sum parity
var<push_constant> pc: Push;

@compute @workgroup_size(64)
fn sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }

    // This dispatch touches only its color.
    if (site_color[site] != pc.color) {
        return;
    }

    let current = spins[site];
    let si = spin_pm(current);
    var nsum: i32 = 0;
    for (var d = 0u; d < N_NEIGHBORS; d = d + 1u) {
        nsum = nsum + spin_pm(spins[neighbors[site * N_NEIGHBORS + d]]);
    }

    // ΔE for the flip, matching the CPU: 2 * s_i * (J * Σ_neighbors + h).
    let dE = 2.0 * f32(si) * (params.j * f32(nsum) + params.h);

    // The conditional probability of landing on the *flipped* state, which is
    // exp(-b dE) / (1 + exp(-b dE)) rearranged to put the exponential in the
    // denominator. That rearrangement is what makes the host's shift by the
    // smallest delta unnecessary here: a strongly downhill flip sends
    // `exp(b dE)` to zero and the probability to one, while a strongly uphill
    // one overflows it to infinity and the probability to zero. Both are the
    // right answers, so the f32 limits are reached from the harmless side and
    // no clamping is needed.
    let p_flip = 1.0 / (1.0 + exp(params.beta * dE));

    // One draw whatever the outcome, and the site is written on that draw alone
    // — there is nothing to reject against, so unlike the Metropolis kernel
    // there is no early-out branch for a downhill move.
    if (keyed_uniform(params.seed, site, pc.sweep) < p_flip) {
        spins[site] = 1u - current;
    }
}
