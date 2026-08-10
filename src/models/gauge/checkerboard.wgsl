// Checkerboard Metropolis sweep for the Z2 gauge model (Q = 2), in any dimension
// admitting a plaquette (D >= 2).
//
// The link counterpart of the Ising `checkerboard.wgsl`. Variables live on links, and
// the unit of interaction is the plaquette rather than the bond, so what two
// threads must avoid is sharing a plaquette, not sharing a bond. Base-site parity
// alone cannot deliver that — two of a plaquette's four links share a base site —
// so a link is colored by its direction *and* its base site's parity: 2D colors,
// and a sweep is 2D dispatches rather than two.
//
// One dispatch updates one color. Threads are launched per site, not per link: a
// dispatch owns one direction of D, so a per-link launch would idle all but one
// thread in 2D, while per-site idles one in two. The link a thread owns is
// `site * D + dir`, since links are packed with direction fastest.
//
// Fixing the direction freezes every link of another direction, and a plaquette
// then holds exactly two links of the pass's direction, whose base sites differ
// by one step and so carry opposite parity. Nothing a thread reads can change
// while it runs.
//
// The base-site parity is read from `site_color` rather than derived here, for
// the reason given in the Ising kernel: the host already holds that table and the
// CPU schedules read it, so uploading it keeps both backends on one coloring and
// leaves no geometry in this kernel. The dimension enters only through `D`, a
// pipeline-overridable constant set when the pipeline is built, from which the
// staple table's shape derives; it is resolved before compilation, so the loops
// below keep constant bounds.
//
// Randomness is counter-based: a link's accept draw is a pure function of
// (seed, link, sweep), so the result is independent of thread order. All passes
// of one sweep share `pc.sweep`; their link sets are disjoint, so no link draws
// twice in a sweep.

override D: u32;

// Plaquettes containing a given link, and the flattened width of its staple row:
// `2(D - 1)` groups of the three other links of each.
override STAPLE_GROUPS: u32 = 2u * (D - 1u);
override STAPLE_STRIDE: u32 = 6u * (D - 1u);

struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    color: u32,   // 0..2D: direction in the high bits, base-site parity in bit 0
};

@group(0) @binding(0) var<storage, read_write> links: array<u32>;    // 0 or 1 per link
@group(0) @binding(1) var<storage, read> staples: array<u32>;        // STAPLE_STRIDE per link
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> site_color: array<u32>;     // coordinate-sum parity
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

    // This dispatch touches only links whose base carries the pass's parity.
    if (site_color[site] != (pc.color & 1u)) {
        return;
    }

    let dir = pc.color >> 1u;
    let link = site * D + dir;

    // Sum over the link's staple groups of the product of the group's three
    // links. The plaquettes containing the link each split into this link's own
    // variable times its staple, so this is the whole of H that depends on it.
    var staple_sum: i32 = 0;
    for (var g = 0u; g < STAPLE_GROUPS; g = g + 1u) {
        let base = link * STAPLE_STRIDE + g * 3u;
        var product: i32 = 1;
        for (var k = 0u; k < 3u; k = k + 1u) {
            product = product * spin_pm(links[staples[base + k]]);
        }
        staple_sum = staple_sum + product;
    }

    // ΔE for the flip, matching the CPU's `-j * ds * staple_sum` at ds = -2σ_l.
    let current = links[link];
    let dE = 2.0 * params.j * f32(spin_pm(current)) * f32(staple_sum);

    let u = keyed_uniform(params.seed, link, pc.sweep);
    if (dE <= 0.0 || u < exp(-params.beta * dE)) {
        links[link] = 1u - current;
    }
}
