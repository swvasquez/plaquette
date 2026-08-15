// The Ising model snippet: the model's side of the device seam. It owns the
// two bindings that are about the model — the neighbor table and the uniform
// block — and the one function the kernel fragments call, `energy_delta`.
// Nothing here mentions a schedule, a kernel, or an axis: the geometry is a
// table, and the walk belongs to the fragments this is assembled with.

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

// The static run parameters. The head — `n_sites`, `seed`, `beta` — is the
// layout contract with the kernel and schedule fragments, which read those
// three fields and nothing else; the tail is this model's own.
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

@group(0) @binding(1) var<storage, read> neighbors: array<u32>;        // 2D per site: +0,-0,+1,-1,...
@group(0) @binding(2) var<uniform> params: Params;

// ΔE for moving site `v` from `current` to `proposed`, matching the CPU:
// `2 * s_v * (J * Σ_neighbors + h)`. At two states the only move is the flip,
// so `proposed` fixes nothing the flip does not; it is part of the seam's
// signature, not of this model's arithmetic.
fn energy_delta(v: u32, current: u32, proposed: u32) -> f32 {
    var nsum: i32 = 0;
    for (var d = 0u; d < N_NEIGHBORS; d = d + 1u) {
        nsum = nsum + spin_pm(vars[neighbors[v * N_NEIGHBORS + d]]);
    }
    return 2.0 * f32(spin_pm(current)) * (params.j * f32(nsum) + params.h);
}
