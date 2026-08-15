// Checkerboard heat bath sweep for the q-state Potts model, in any dimension.
//
// The sibling of this model's `checkerboard.wgsl`, sharing its bindings, its
// bounds check, its color guard and its `offset_for` unpacking. The coloring
// belongs to the schedule rather than the kernel, so the independence argument
// made there carries over unchanged and is not repeated.
//
// This is the one of the three heat bath kernels that is more than a different
// ending. The Metropolis kernel picks a candidate first and then needs a single
// signed counter, since only two labels can matter to it. The heat bath commits
// to no candidate, so it needs how many neighbors carry *each* label, and then a
// weight per label to draw against. `docs/heat-bath.md` describes the update.
//
// Two tokens are substituted before this compiles rather than one. `$H_VECTORS$`
// is inherited from the Metropolis kernel and is explained there. `$Q$` is new
// and is needed for the same underlying reason: the tallies below are arrays
// whose length is the state count, and a WGSL array length must be a
// const-expression, which a pipeline `override` is not. Both are filled in by
// `potts_gpu::shader_for`.
//
// Randomness stays counter-based on `(seed, site, sweep)`. The heat bath draws
// once where the Metropolis kernel draws twice, since there is no separate
// proposal to key — so this kernel does not use `PROPOSAL_KEY` at all.

override D: u32;

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

// The state count, substituted by the host. Unlike `params.q`, which is a
// runtime uniform the Metropolis kernel is content with, this has to be a
// constant because it sizes the arrays below.
const Q_STATES: u32 = $Q$u;

// How many vec4 slots the per-label offsets occupy — see `checkerboard.wgsl`.
const H_VECTORS: u32 = $H_VECTORS$u;

struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
    q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    h: array<vec4<f32>, H_VECTORS>,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    color: u32,   // which color this dispatch updates (0 or 1)
};

@group(0) @binding(0) var<storage, read_write> labels: array<u32>;     // 0..q per site
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;        // 2D per site: +0,-0,+1,-1,...
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> site_color: array<u32>;       // coordinate-sum parity
var<push_constant> pc: Push;

// One label's energy offset, unpacked from the vec4 it was folded into.
fn offset_for(label: u32) -> f32 {
    return params.h[label / 4u][label % 4u];
}

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

    // How many neighbors carry each label. One pass over the neighbor row
    // serves every candidate, which is what keeps this kernel at the same
    // memory traffic as the Metropolis one despite pricing q states instead
    // of one.
    var counts: array<u32, Q_STATES>;
    for (var s = 0u; s < Q_STATES; s = s + 1u) {
        counts[s] = 0u;
    }
    for (var d = 0u; d < N_NEIGHBORS; d = d + 1u) {
        let neighbor = labels[neighbors[site * N_NEIGHBORS + d]];
        counts[neighbor] = counts[neighbor] + 1u;
    }

    // Each candidate priced against the current label, matching the CPU's
    // ΔE = -j * (agree_after - agree_before) - (h[after] - h[before]). The
    // current label's own entry comes out exactly zero, so the running minimum
    // can start there.
    let current = labels[site];
    let agree_now = f32(counts[current]);
    let offset_now = offset_for(current);
    var weights: array<f32, Q_STATES>;
    var lowest = 0.0;
    for (var s = 0u; s < Q_STATES; s = s + 1u) {
        let dE = -params.j * (f32(counts[s]) - agree_now) - (offset_for(s) - offset_now);
        weights[s] = dE;
        lowest = min(lowest, dE);
    }

    // Shifting by the smallest delta puts the largest exponent at zero, so the
    // biggest weight is exactly one and nothing overflows. The Ising and gauge
    // heat baths get away without this because at two states the ratio can be
    // written so that both limits saturate correctly; with a sum over q terms
    // there is no such rearrangement, so the shift is load-bearing here.
    var total = 0.0;
    for (var s = 0u; s < Q_STATES; s = s + 1u) {
        weights[s] = exp(-params.beta * (weights[s] - lowest));
        total = total + weights[s];
    }

    // One draw, walked against the unnormalized cumulative sum. The last label
    // is the structural fallback rather than a case — it takes every draw the
    // earlier ones do not claim — so rounding in `total` cannot leave the walk
    // without an answer, and no clamp is needed the way the Metropolis kernel's
    // proposal draw needs one. The name avoids `target`, which WGSL reserves.
    let threshold = keyed_uniform(params.seed, site, pc.sweep) * total;
    var chosen = Q_STATES - 1u;
    var cumulative = 0.0;
    for (var s = 0u; s + 1u < Q_STATES; s = s + 1u) {
        cumulative = cumulative + weights[s];
        if (threshold < cumulative) {
            chosen = s;
            break;
        }
    }

    labels[site] = chosen;
}
