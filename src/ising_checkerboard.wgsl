// Checkerboard Metropolis sweep for the Ising model (Q = 2), in any dimension.
//
// One dispatch updates one color: a thread per site, acting only on sites whose
// color matches `pc.color`. Within a color no two sites are neighbors, so every
// thread reads only the *other* color — untouched during this dispatch — and
// there is no read/write conflict. A full sweep is two dispatches, color 0 then
// color 1, with the pass boundary as the barrier.
//
// The color is read from `site_color` rather than derived here. It is the
// coordinate-sum parity, which the host already has as a table on the lattice
// and which the CPU schedules read from that same table; uploading it keeps the
// two backends agreeing on the coloring by construction instead of by two
// implementations happening to match, and it is what leaves this kernel with no
// geometry in it at all. Nothing here mentions an axis or an extent.
//
// The dimension enters only through `D`, a pipeline-overridable constant set
// when the pipeline is built, from which the neighbor row width derives. It is
// resolved before the shader is compiled, so the loop below keeps a constant
// bound.
//
// Randomness is counter-based: a site's accept draw is a pure function of
// (seed, site, sweep), so the result is independent of thread order and does not
// depend on a shared stream. Both color passes of one sweep share `pc.sweep`;
// their site sets are disjoint, so no site draws twice in a sweep.

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

    let u = keyed_uniform(params.seed, site, pc.sweep);
    if (dE <= 0.0 || u < exp(-params.beta * dE)) {
        spins[site] = 1u - current;
    }
}
