// Checkerboard Metropolis sweep for the q-state Potts model, at any Q and in any
// dimension.
//
// This file is a *template*, not a finished shader: the host substitutes
// `$H_VECTORS$` before compiling it. That is the one thing here that cannot be
// handled the way the dimension is — `D` arrives as a pipeline-overridable
// constant, but an override cannot size an array in the uniform address space,
// and the per-label offsets are such an array.
//
// One dispatch updates one color: a thread per site, acting only on sites whose
// color matches `pc.color`. Within a color no two sites are neighbors, so every
// thread reads only the *other* color — untouched during this dispatch — and
// there is no read/write conflict. A full sweep is two dispatches, color 0 then
// color 1, with the pass boundary as the barrier. That argument reads only the
// adjacency, so it is the same one the Ising kernel rests on; the number of
// states does not enter it.
//
// The color is read from `site_color` rather than derived here, as in the Ising
// kernel: the host already holds the coordinate-sum parities as a table and the
// CPU schedules read that same table, so uploading it keeps the two backends
// agreeing on the coloring by construction. Nothing here mentions an axis or an
// extent.
//
// Two things differ from the Ising kernel, and both follow from the labels being
// unordered. A site stores a label in `0..q` rather than a spin, so the energy
// term is a *count* of matching neighbors rather than a sum of signed products.
// And the proposal has `q - 1` candidates rather than one, so it must draw:
// where the Ising kernel flips, this picks uniformly among the labels that are
// not the current one, which is what keeps the proposal symmetric and so lets
// the acceptance rule below drop the Hastings ratio.
//
// `q` arrives in the uniform block rather than as an `override`, unlike the
// dimension. `D` fixes the neighbor row width, which has to be a constant for
// the loop below to keep a constant bound; `q` only scales a draw and bounds
// nothing, so there is nothing to gain from resolving it at pipeline build time.
//
// Randomness is counter-based: a site's two draws are pure functions of
// (seed, site, sweep), so the result is independent of thread order and does not
// depend on a shared stream. Both color passes of one sweep share `pc.sweep`;
// their site sets are disjoint, so no site draws twice in a sweep.

override D: u32;

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

// Keys the proposal draw off a different stream from the accept draw. The two
// must be independent — a site that picked its candidate and its acceptance
// from the same number would accept exactly the labels the mixing happens to
// send low. Any constant unrelated to the ones inside `keyed_uniform` does the
// job, since the seed is mixed before it is used.
const PROPOSAL_KEY: u32 = 0x85ebca6bu;

// How many vec4 slots the per-label offsets occupy: four labels to a slot,
// rounded up. WGSL requires a uniform array's length to be written literally in
// the source, and the source has no way to see the host's state count — so the
// host substitutes this one token before compiling (`potts_gpu::shader_for`),
// which is why this file is a template rather than a finished shader.
//
// Packing four offsets into a vec4 rather than declaring `array<f32, N>` is
// what avoids WGSL's sixteen-byte stride on uniform array elements, which would
// otherwise waste three words in four.
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

    let current = labels[site];

    // Draw uniformly among the `q - 1` labels that are not the current one, then
    // skip past the current index so the map onto them is a bijection. The clamp
    // is not paranoia: `keyed_uniform` returns the largest float below one, and
    // scaling that by `q - 1` can round *up* to exactly `q - 1` once `q - 1`
    // exceeds two, which would put the proposal one past the last label.
    let alternatives = params.q - 1u;
    var draw = u32(keyed_uniform(params.seed ^ PROPOSAL_KEY, site, pc.sweep) * f32(alternatives));
    if (draw >= alternatives) {
        draw = alternatives - 1u;
    }
    var proposed = draw;
    if (proposed >= current) {
        proposed = proposed + 1u;
    }

    // ΔE = -j * (agree_after - agree_before) - (h[proposed] - h[current]),
    // matching the CPU. Only the bonds touching this site change, and each
    // counts how many neighbors carry the proposed or the current label; the two
    // labels differ, so a neighbor lands in at most one of the counts. The
    // offset term reads only this site's own two entries, which is why a
    // constant added to every entry cancels here as it does on the host.
    var change: i32 = 0;
    for (var d = 0u; d < N_NEIGHBORS; d = d + 1u) {
        let s = labels[neighbors[site * N_NEIGHBORS + d]];
        if (s == proposed) {
            change = change + 1;
        }
        if (s == current) {
            change = change - 1;
        }
    }
    let offset = offset_for(proposed) - offset_for(current);
    let dE = -params.j * f32(change) - offset;

    let u = keyed_uniform(params.seed, site, pc.sweep);
    if (dE <= 0.0 || u < exp(-params.beta * dE)) {
        labels[site] = proposed;
    }
}
