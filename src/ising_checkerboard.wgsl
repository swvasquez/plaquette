// Checkerboard Metropolis sweep for the 2D Ising model (Q = 2).
//
// One dispatch updates one color: a thread per site, acting only on sites whose
// coordinate-sum parity matches `pc.color`. Within a color no two sites are
// neighbors, so every thread reads only the *other* color — untouched during
// this dispatch — and there is no read/write conflict. A full sweep is two
// dispatches, color 0 then color 1, with the pass boundary as the barrier.
//
// Randomness is counter-based: a site's accept draw is a pure function of
// (seed, site, sweep), so the result is independent of thread order and does not
// depend on a shared stream. Both color passes of one sweep share `pc.sweep`;
// their site sets are disjoint, so no site draws twice in a sweep.

struct Params {
    n_sites: u32,
    width: u32,   // extent along axis 0, for the parity coloring
    seed: u32,
    _pad0: u32,
    beta: f32,
    j: f32,
    h: f32,
    _pad1: f32,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    color: u32,   // which color this dispatch updates (0 or 1)
};

@group(0) @binding(0) var<storage, read_write> spins: array<u32>;      // 0 or 1 per site
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;        // 4 per site: +0,-0,+1,-1
@group(0) @binding(2) var<uniform> params: Params;
var<push_constant> pc: Push;

@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.n_sites) {
        return;
    }

    // Coordinate-sum parity: this dispatch touches only its color.
    let x0 = site % params.width;
    let x1 = site / params.width;
    if (((x0 + x1) & 1u) != pc.color) {
        return;
    }

    let current = spins[site];
    let si = spin_pm(current);
    var nsum: i32 = 0;
    for (var d = 0u; d < 4u; d = d + 1u) {
        nsum = nsum + spin_pm(spins[neighbors[site * 4u + d]]);
    }

    // ΔE for the flip, matching the CPU: 2 * s_i * (J * Σ_neighbors + h).
    let dE = 2.0 * f32(si) * (params.j * f32(nsum) + params.h);

    let u = keyed_uniform(params.seed, site, pc.sweep);
    if (dE <= 0.0 || u < exp(-params.beta * dE)) {
        spins[site] = 1u - current;
    }
}
